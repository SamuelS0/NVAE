import json

d = open('crmnist.json','r')

data = json.load(d)

for i in data:
    print(i)

for i in data['domain_data']:
    print(i + ': type :' + str(type(i)))

val = 10

print(f"val is {{val}}")