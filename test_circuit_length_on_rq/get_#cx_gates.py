import matplotlib.pyplot as plt
dic = {}
with open("qiskit.txt") as file:
    for item in file:
        if item.__contains__('circuit.cx'):

            print(item)
            print(len(item))
            if len(item)==35:
                key = item[18]+item[19] + '-' + item[30]+item[31]
                print(key)
            if len(item) == 33:
                print(item[18]+'-'+item[29])
                key = item[18]+'-'+item[29]
            if len(item) == 34 and item[19].isdigit():
                key = item[18] + item[19] + '-' + item[30]
                print(key)
            if len(item) == 34 and not item[19].isdigit():
                key = item[18] + '-' + item[29]+item[30]
                print(key)
            if key in dic.keys():
                dic[key]+=1
            else:
                dic[key]=1

print(dic)
print(sum(dic.values()))

dic2={}
for i, key  in enumerate(dic.copy()):

    for j, key2  in enumerate(dic.copy()):
        if key != key2 and len(key)==len(key2):
            if len(key) == 3 and key[0] == key2[2] and key[2] == key2[0]:
                dic2[key + ' and ' + key2] = dic[key] + dic[key2]
                del dic[key]
                del dic[key2]

            if len(key) == 5 and key[0] + key[1] == key2[3] + key2[4] and key[3] + key[4] == key2[0] + key2[1]:
                dic2[key + ' and ' + key2] = dic[key] + dic[key2]
                del dic[key]
                del dic[key2]
            if len(key) == 4 and not key[1].isdigit() and key[0] == key2[3] and key[2] + key[3] == key2[0] + key2[1]:
                dic2[key + ' and ' + key2] = dic[key] + dic[key2]
                del dic[key]
                del dic[key2]
            if len(key) == 4 and key[1].isdigit() and key[0] + key[1] == key2[2] + key2[3] and key[3] == key2[0]:
                dic2[key + ' and ' + key2] = dic[key] + dic[key2]
                del dic[key]
                del dic[key2]




print(dic2)
print(sum(dic2.values()))
z = {**dic2, **dic}
print(z)
print(sum(z.values()))
# temp=dic.copy()
# print(temp)
# dic2=dic2.update(temp)
# print(dic2)
def draw_from_dict(dicdata,RANGE, heng=0):

    fig, ax = plt.subplots(figsize=(10, 5), dpi=90)
    by_value = sorted(dicdata.items(),key = lambda item:item[1],reverse=True)
    x = []
    y = []
    for d in by_value:
        x.append(d[0])
        y.append(d[1])
    if heng == 0:
        bars = ax.bar(x[0:RANGE], y[0:RANGE])
        ax.bar_label(bars)
        plt.show()
        return
    elif heng == 1:
        bars= ax.barh(x[0:RANGE], y[0:RANGE])
        ax.bar_label(bars)
        plt.show()
        return
    else:
        return "heng的值仅为0或1！"
draw_from_dict(z,len(z), 1)



