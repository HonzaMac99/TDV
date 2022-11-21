class TestClass:

    def __init__(self, name, age):
        self.age = age
        self.name = name
    
    def say_hello(self):
        print("Hi, I am", self.name, "and I am", self.age, "years old")

class TestClass2(TestClass):
    
    def say_hello(self):
        print("Hello, I am", self.name, "and I am", self.age, "years old")

my_new_object = TestClass("Peter", 18)
my_new_object2 = TestClass2("Philip", 34)

my_new_object.say_hello()
my_new_object2.say_hello()
