from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")




print("\n\nTraining XOR\n\n")
xor_training_data =  [
    ([.9,.8,.6,.3,.1],[1]),
    ([.8,.8,.4,.6,.4],[1]),
    ([.7,.2,.4,.6,.3],[1]),
    ([.5,.5,.8,.4,.8],[0]),
    ([.3,.1,.6,.8,.8],[0]),
    ([.6,.3,.4,.3,.6],[0])
]
test_data = [
    ([1,1,1,.1,.1]),
    ([.5,.2,.1,.7,.7]),
    ([.8,.3,.3,.3,.8]),
    ([.8,.3,.3,.8,.3]),
    ([.9,.8,.8,.3,.6])
]
 
nn = NeuralNet(5, 5, 1)
nn.train(xor_training_data)
print(nn.test_with_expected(xor_training_data))
 
 
for input_data in test_data:
    prediction = nn.evaluate(input_data)  # Get the network's prediction for each test input
    print(f"Input: {input_data} -> Predicted: {prediction[0]}")
 
     
print('the following is ih weights'
 )
print(nn.get_ih_weights())
print('the following is ho weights')
print(nn.get_ho_weights())
 
 
 
 
# xor_testing_data = [
#     ((0.0, 0.0)),
     
# ]
# nn = NeuralNet(2,2,1)
# nn.train(xor_training_data)
 
# print(nn.test(xor_testing_data))
 
# test_training_data =  [
#     ([.9,.8,.6,.3,.1],[1]),
#     ([.8,.8,.4,.6,.4],[1]),
#     ([.7,.2,.4,.6,.3],[1]),
#     ([.5,.5,.8,.4,.8],[0]),
#     ([.3,.1,.6,.8,.8],[0]),
#     ([.6,.3,.4,.3,.6],[0])
# ]
# test_data = [
#     ([1,1,1,.1,.1]),
#     ([.5,.2,.1,.7,.7]),
#     ([.8,.3,.3,.3,.8]),
#     ([.8,.3,.3,.8,.3]),
#     ([.9,.8,.8,.3,.6])
# ]
