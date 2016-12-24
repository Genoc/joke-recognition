import urllib2
import json
import os

ret = urllib2.urlopen("http://api.icndb.com/jokes/random").read()
joke = json.loads(ret)

joke_string = joke["value"]["joke"]
print(joke_string)
joke_string2 = joke_string.replace("'","").replace("&quot;","")
print(joke_string.replace("'","").replace("&quot;",""))

os.system('say "' + joke_string2 + '"')
