from NN import *

model = MNISTModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
epochs = 1
train_model(model, optimizer, loss_fn, train_dataloader, epochs)
model.eval()
with torch.inference_mode():
    acc_sum = 0
    for batch, (x, y) in enumerate(test_dataloader):
        acc_sum += torch.sum(torch.eq(torch.argmax(nn.Softmax(dim=1)(model(x)), dim = 1), y))
    test_acc = acc_sum/(len(test_data))

print(f"Test accuracy --> {test_acc*100:.2f}%")
