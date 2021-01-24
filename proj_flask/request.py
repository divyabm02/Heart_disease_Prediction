import requests

# URL
url = 'http://localhost:5000/prediction.html/'

# Change the value of experience that you want to test
payload = {
	'age':40,"edu":3,"CS":0,"cpd":0,"BPM":0,"PS":0,"PH":1,"D":0,"TC":195,"sysBP":106,"diaBP":81,"BMI":24.7,"HR":78,"glu":77.0
}

r = requests.post(url,json=payload)

print(r.json())