#inputVar=[[1,2,3,4],[1,2,3,4]]

print("inputVar---------->",inputVar)

if type(inputVar) == int:
   # output="The variable is int ",inputVar
   output=inputVar

elif type(inputVar)==float:
    #output="The variable is float",inputVar
    output=inputVar
        
elif type(inputVar) == str:
    output="The variable is string"
elif len(inputVar)>=1:
   output="The variable is array",inputVar
   #output=inputVar
else:
    #output="Could not identify the variable", inputVar
    output=inputVar
print("output Variable:",output)

