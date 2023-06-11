from NN import *
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(42)
model = FashionMNISTModel(); model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 1

#test_accuracy = torch.eq(torch.argmax(nn.Softmax(dim=1)(model(test_data)), dim=1))


train_model(model, optimizer, loss_fn, train_dataloader, epochs)

model.eval()
with torch.inference_mode():
    acc_sum = 0
    for batch, (x, y) in enumerate(test_dataloader):
        acc_sum += torch.sum(torch.eq(torch.argmax(nn.Softmax(dim=1)(model(x)), dim=1), y))
    print(f"Accuracy --> {(acc_sum/10000)*100:.2f}%")



def predict(index):
    image, label = test_data[index]
    model.eval()
    with torch.inference_mode():
        pred = torch.argmax(nn.Softmax(dim=1)(model(image)), dim=1)

    print(pred)
    print(f"Predicted --> {pred}; {test_data.classes[pred]}")
    print(f"Expected --> {label}; {test_data.classes[label]}")
    print()
    plt.imshow(image.squeeze(), cmap="gray")
    plt.show()
    print("Try a different index (max = 9999): ")
    index = int(input())
    return predict(index)
predict(0)


