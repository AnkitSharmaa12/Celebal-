# Printing the Lower triangle by using the symbol '*' .
print("Assignment 1\n")
a=1
b=5
print("Printing the Lower triangle")
while(a<=b):
    c=1
    while(c<=a):
        print('*',end=' ')
        c=c+1
    print()
    a=a+1

print()


# Printing the upper triangle by using the symbol '*'.

d=1
e=5
print("Printing the Upper triangle\n")
while (d<=e):
    space = 1
    while (space<d):
        print(" ",end=' ')
        space=space+1
    f=d
    while(f<=e):
        print('*',end=' ')
        f +=1
    print()
    d +=1

print()

# Printing the Pyramid with the help of the symbol '*'.

print("Printing the Pyramid from '*'\n ") 
i=1
n=5
while(i<=n):
    space=n-i
    while space:
        print(" ",end=' ')
        space -=1
    
    j=1
    while(j<=i):
        print('*',end=' ')
        j +=1

    k=i-1
    while(k):
        print('*',end=' ')
        k -=1 

    print()
    i +=1
