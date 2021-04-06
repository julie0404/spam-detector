import json
import boto3
import re
import os
from datetime import datetime
import dateutil
from spam_classifier import one_hot_encode
from spam_classifier import vectorize_sequences

vocabulary_length = 9013
input_date_format = '%a, %d %b %Y %H:%M:%S %z'
output_date_format = '%m-%d-%Y %H:%M:%S %Z %z'
EST = dateutil.tz.gettz('US/Eastern')

sagemaker_endpoint = "sms-spam-classifier-mxnet-2021-04-05-19-57-00-902"

def spam_classify(mail_body):
    mail_body = mail_body.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    mail_body = [mail_body]
    runtime= boto3.client('runtime.sagemaker')
    one_hot_test_messages = one_hot_encode(mail_body, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    msg = json.dumps(encoded_test_messages.tolist())
    response = runtime.invoke_endpoint(EndpointName=sagemaker_endpoint,ContentType='application/json', Body=msg)
    result = response['Body']
    res = json.loads(result.read().decode("utf-8"))
    
    predicted_score = int(res['predicted_label'][0][0])
    predicted_probability = float(res['predicted_probability'][0][0])
    predicted_label = 'Spam' if predicted_score == 1 else 'Not Spam'
    predicted_probability = predicted_probability if predicted_score == 1 else (1 - predicted_probability)
    return predicted_label, predicted_probability 
    
    
def send_response(sent_date, sender_email, subject, mail_body, prediction, score):
    score = score * 100
    response_mail = "Your email was received at %s with the subject \"%s\".\n\nHere is the email body:\n\n%s\n\nThe email was classified as %s with a %.2f%% confidence." % (sent_date, subject, mail_body[0:100], prediction, score)
    client = boto3.client('ses')
    response = client.send_email(Source='test123@6998hw3.com', 
        Destination={'ToAddresses': [sender_email]},
        Message={
            'Subject': {
                'Data': subject,
                'Charset': 'utf-8'
            },
            'Body': {
                'Text': {
                    'Data': response_mail,
                    'Charset': 'utf-8'
                }
            }
        }
    )


def clean_body(body):
    regex_rule = re.compile(r"(Feedback-ID:|Content-Transfer-Encoding:|Content-Type: *text/plain) *[^\r\n]*(.+)", re.DOTALL)
    mail_body = re.search(regex_rule, body).group(2)
    check1 = re.search(r"^\s*Content-Transfer-Encoding: *[^\r\n]*(.+)", mail_body, re.DOTALL)
    if check1:
        mail_body = check1.group(1)
    check2 = re.search(r"(.+)\r\n--[0-9a-zA-Z]+\r\nContent-Type:.*", mail_body, re.DOTALL)
    if check2:
        mail_body = check2.group(1)
    mail_body = mail_body.replace('=\r\n', '').replace('=E2=80=99', '\'').replace('\r\n\r\n', '\r\n').strip()
    return mail_body


def cook_mail(bucket, key):
    client = boto3.client('s3')
    response = client.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read()
    body = body.decode('utf8')
    sent_date = re.search('Date: (.*) *[\r\n]', body).group(1).strip()
    sender_email = re.search('From: (.*) *[\r\n]', body).group(1).strip()
    check1 = re.search('[^<]* *<(.*@.*)>', sender_email)
    if check1:
        sender_email = check1.group(1)
    subject = re.search('Subject: (.*) *[\r\n]', body).group(1).strip()
    sent_date = datetime.strptime(sent_date, input_date_format).astimezone(EST).strftime(output_date_format)
    mail_body = clean_body(body)
    return sent_date, sender_email, subject, mail_body


def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    sent_date, sender_email, subject, mail_body = cook_mail(bucket, key)
    prediction, score = spam_classify(mail_body)
    response = send_response(sent_date, sender_email, subject, mail_body, prediction, score)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Mail successfully sent')
    }
