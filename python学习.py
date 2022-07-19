#数据类型：
#Numbers（数字）
a=233
#String（字符串）
a="233"
#List（列表）
a=[2,3,[4,5]]
#Tuple（元组） 元组不能二次赋值
a=(2,3,4)
#Dictionary（字典） 相当于map
a={"a":1,"b":2}
a['c']=3

#逻辑运算符
and
or
not

#成员运算符
in
not in
s=[1,2,3]
if 1 in s:
	print("233")

#身份运算符  (判断两个标识符是不是引用自一个对象,一般用于列表、元组、字典)
#数字、字符串是不可变类型，即value(值)一旦改变，id（内存地址）也改变，而相同的value对应相同的id
#列表、元组、字典是可变类型，即在id(内存地址)不变的情况下，value（值）可以变
is
is not

#if else(注意缩进)
if 判断条件1:
    执行语句1……
elif 判断条件2:
    执行语句2……
elif 判断条件3:
    执行语句3……
else:
    执行语句4……

#while(注意缩进)
while 判断条件(condition)：
    执行语句(statements)……
	continue
	break

#for
for iterating_var in sequence:
   statements(s)
例1：
fruits = ['banana', 'apple',  'mango']
for index in range(len(fruits)):
   print ('当前水果 : %s' % fruits[index])
例2：
for num in range(10,20):  # 迭代 10 到 20 之间的数字
   for i in range(2,num): # 根据因子迭代
      if num%i == 0:      # 确定第一个因子
         j=num/i          # 计算第二个因子
         print ('%d 等于 %d * %d' % (num,i,j))
         break            # 跳出当前循环
   else:                  # 循环的 else 部分
      print ('%d 是一个质数' % num)