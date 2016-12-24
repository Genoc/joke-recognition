import urllib2
import json

ret = urllib2.urlopen("http://api.icndb.com/jokes/random").read()
joke = json.loads(ret)

joke_string = joke["value"]["joke"]

print(joke_string)