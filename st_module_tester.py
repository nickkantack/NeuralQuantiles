
import torch
from linear_st import LinearST

def main():

    in_features = 10
    out_features = 5
    batch_size = 8

    model = LinearST(sim_queue_length=100, quantile_count=4, in_features=in_features, out_features=out_features)

    for i in range(10):
        input_tensor = torch.rand(batch_size, in_features)
        _ = model(input_tensor)

    print(model.quantiles)

    print("done")

if __name__ == "__main__":
    main()
