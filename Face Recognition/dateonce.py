from datetime import datetime
date=datetime.now()
strr=""
strr=datetime.isoformat(date)
st=strr[0:16]
with open("date.txt","w")as d:
    d.write(strr)
    d.close()
b=set(open("attend.txt"))
c=str(b)
with open ("mark.txt","w")as f:
    f.write(st)
    f.write(c)
    f.close()
