#Design a python program to accept a file name through command line arguments.

import sys
import re
with open(sys.argv[1]) as filename:
        s=''
        for line in filename:
               s+=line;
#1. Print all currencies in text, Accepted- $, ₹, £ 

p1=u"[$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6]"
print(re.findall(p1,s))

#2. Print all date times in the text- dd/mm/yyyy, dd/mm/yy, mm/dd/yyyy, mm/dd/yy

p2='[0-9][0-9]/[0-9][0-9]/[0-9]*'
print(re.findall(p2,s))

#3. Print all cardinilities and orders- 4th, fifth, sixth, 1st, 2nd, nineteenth, fifth

p3='[0-9]+ ?[st|nd|rd|th]+|first|second|third|fifth|sixth|nineteenth'
print(re.findall(p3,s))

#4. Print all 4 letter words that begin with vowels

p4='[ \\n][aeiouAEIOU][a-zA-Z][a-zA-Z][a-zA-Z][ \\n]'
print(re.findall(p4,s))
