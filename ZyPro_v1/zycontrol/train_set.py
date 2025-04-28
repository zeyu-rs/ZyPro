import torch 
from datetime import datetime
import os
import torch.optim.lr_scheduler as lr_scheduler
from zycontrol.training import train_one_epoch_seg

def train_model_seg(model, dataloader, criterion, optimizer, scheduler, log_path, task_name, num_epochs, device, save_interval=10):
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    out_dir = os.path.join(log_path, now_time)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    log_file = os.path.join(out_dir, task_name + now_time + '.txt')
    
    with open(log_file, 'w') as file:
        pass
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    for epoch in range(num_epochs):
        loss = train_one_epoch_seg(model, dataloader, criterion, optimizer, device)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        log_str = f"Training... Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, LR: {lr}"

        with open(log_file, 'a') as file:
            now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file.write(task_name + '_' + now_time + '_' + log_str + '\n')

        print(log_str)

        # 每 10 轮保存一次模型
        if (epoch + 1) % save_interval == 0:
            pth_file = os.path.join(out_dir, f"{task_name}_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), pth_file)
            print(f"Model saved at {pth_file}")

    # 训练结束后保存最终模型
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    final_pth_file = os.path.join(out_dir, f"{task_name}_final_{now_time}.pth")
    torch.save(model.state_dict(), final_pth_file)
    print(f"Final model saved at {final_pth_file}")
