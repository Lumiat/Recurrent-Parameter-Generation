import os
import sys
if __name__ == "__main__":
    from train import *
else:  # relative import
    from .train import *



# get checkpoint path
try:
    test_item = sys.argv[1]
except IndexError:
    assert __name__ == "__main__"
    test_item = "./checkpoint"

# get log file path
try:
    result_file = sys.argv[2]
except IndexError:
    result_file = None

test_items = []
if os.path.isdir(test_item):
    for item in os.listdir(test_item):
        item = os.path.join(test_item, item)
        test_items.append(item)
elif os.path.isfile(test_item):
    test_items.append(test_item)


for item in test_items:
    state = torch.load(item, map_location="cpu")
    model.load_state_dict({key: value.to(torch.float32).to(device) for key, value in state.items()})
    loss, acc, all_targets, all_predicts = test(model=model)
    
    # save result if result_file exists
    if result_file:
        result_data = {
            "loss": float(loss),
            "acc": float(acc),
            "model_path": item
        }
        result_dir = os.path.dirname(result_file)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Results saved to {result_file}", flush=True)