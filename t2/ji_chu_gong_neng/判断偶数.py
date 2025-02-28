#输出整除函数
def i1(n):
    return n%2==0

n=input("输入：")
while n!='end':
    while n!='end':
        n=int(n)
        if i1(n):
            print(f'{n}为偶数')
            break
        else:
            print(f'{n}为奇数')
            break
    n = input("输入：")
else:
    print('end')
