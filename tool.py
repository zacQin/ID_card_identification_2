import re

# s = '\r\nabc\t123\nxyz'
s = '{广州市荔湾区和平东\路92号地下'
#
# print(re.sub('[\r\n\t\s+\.\!\/_,$%^*(+\"\'+|+——！，。？、~@#￥%……{}&*（）]', '', s))
# print(re.sub('[\r\n\t\s`~!@#$%^&*()_+={}:";<>,.?/·～！@#¥%……&*（）——+=【】「」；："《，》。？/]', '', s))

def clean_symbol(str):
    str = re.sub('[\r\n\t\s\\\`~!@#$%^qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM&○’□*()_+={}:";<>,.?|、/·～！@#¥%……&*（）——+=【】「」；："《，》。？/]', '', str)
    return str
