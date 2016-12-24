import urllib2
import json
import os

ret = urllib2.urlopen("http://api.icndb.com/jokes/random").read()
joke = json.loads(ret)

joke_string = joke["value"]["joke"]

print(joke_string)
os.system('say Okay. Here is a joke.')
os.system('say '+ joke_string)