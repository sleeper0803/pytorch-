# -*- coding: utf-8 -*-
'''
#测试函数
def function1 (string):
    print("your name:",string)
    return
def function4 (arg1,*arg2):
    print(arg1)
    for i in arg2:
        print(i)
    return
 
function4(10,1,4,5,6,7)
#a = function1("1")
#print(a)
#function1("hello")
'''


'''
#测试类和继承类
class MyStudent(object):
    Student_count=0
    
    def __init__(self,name,age):
        self.name=name
        self.age=age
        MyStudent.Student_count+=1
        
    def dis_student(self):
        print("student name:",self.name)
        
class Newstudent(MyStudent):
    def __init__(self, name, age):
        super().__init__(name, age)

    def dis_student(self):
       print("mm student1 name:",self.name)

student1 = MyStudent("Meng","20")
student1.dis_student()

student2 = Newstudent("Meng","20")
student2.dis_student()
'''


'''
#测试dot
import torch
import numpy as np
a=torch.arange(20).reshape(5,4)
b=torch.arange(4).reshape(4,1)

c=np.dot(a,b)
#d=torch.dot(a,b)
print(c)
'''


'''
#测试矩阵和向量
import torch
A1=torch.arange(5.0).reshape(5,1)
A2=torch.arange(5.0)
e=torch.arange(10.0).reshape(2,5)

D1=torch.mm(e,A1)
D2=torch.mv(e,A2)
#A1是矩阵，而A2只是向量
print(D1)
print(D2)
'''

'''
#类继承的事例
class SchoolMember(object):
    #学习成员基类
    member = 0
 
    def __init__(self, name, age, sex):
        self.name = name
        self.age = age
        self.sex = sex
        self.enroll()
 
    def enroll(self):
        '注册'
        print('just enrolled a new school member [%s].' % self.name)
        SchoolMember.member += 1
 
    def tell(self):
        print('----%s----' % self.name)
        for k, v in self.__dict__.items():
            print(k, v)
        print('----end-----')
 
    def __del__(self):
        print('开除了[%s]' % self.name)
        SchoolMember.member -= 1

class Teacher(SchoolMember):
    '教师'
    def __init__(self, name, age, sex, salary, course):
        SchoolMember.__init__(self, name, age, sex)
        self.salary = salary
        self.course = course
 
    def teaching(self):
        print('Teacher [%s] is teaching [%s]' % (self.name, self.course))
 
class Student(SchoolMember):
    '学生'
 
    def __init__(self, name, age, sex, course, tuition):
        SchoolMember.__init__(self, name, age, sex)
        self.course = course
        self.tuition = tuition
        self.amount = 0
 
    def pay_tuition(self, amount):
        print('student [%s] has just paied [%s]' % (self.name, amount))
        self.amount += amount
 
t1 = Teacher('Wusir', 28, 'M', 3000, 'python')
t1.tell()
s1 = Student('haitao', 38, 'M', 'python', 30000)
s1.tell()
s2 = Student('lichuang', 12, 'M', 'python', 11000)
print(SchoolMember.member)
del s2
 
print(SchoolMember.member)
'''

'''
#测试backward中的参量
import torch 
from torch.autograd import Variable 
x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 1
#y.backward()
x2=Variable(torch.arange(5.0),requires_grad=True)
y2=Variable(torch.arange(5.0),requires_grad=True)
m2=torch.ones_like(y2)

y2.backward(m2)
with torch.no_grad():
    print(y2.grad.data)
    print(y2.data)
#m2.backward()
a=10
'''
'''
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module): 
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)# submodule: Conv2d
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
'''
a = ["dd","aa"]

aaa= next(iter(a))

m=10

from torch.nn import Module
from torch.nn.modules import module
class model(module):
    def __init__(self) -> None:
        super().__init__()


Model =model()

Model.cuda()
m=10