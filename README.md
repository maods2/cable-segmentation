# cable-segmentation

## Clone repository
```
git clone https://github.com/maods2/cable-segmentation.git
cd cable-segmentation
```

## Using tmux
If you want to run the training script on a remote server, consider using tmux to manage your terminal session. This allows you to easily reconnect to your session if your connection to the remote server is interrupted.
```
# Install tmux if not installed yet
sudo apt-get update && sudo apt-get install tmux

# start tmux session
tmux new -s <my-session>

# If you already have a tmux session, attach it with:
tmux attach -t <my-session>
```
### Other tmux Commands

1. **Enter Copy Mode**: Press `Ctrl + b`, then `[`. This allows you to scroll through the terminal output and select text.
  
2. **Navigate in Copy Mode**:
   - **Scroll Up**: Use the `Up Arrow` key or press `Ctrl + b` followed by `Page Up`.
   - **Scroll Down**: Use the `Down Arrow` key or press `Ctrl + b` followed by `Page Down`.

3. **Exit Copy Mode**: Press `q` or `Enter` to return to normal mode.


## Provide executable permission
```
chmod +x run_gpu_container.sh
```

## Run the GPU container
```
./run_gpu_container.sh
```

## Download dataset
```
make download-dataset
```

## Conda save env command
```
conda env export --no-builds > environment.yml
```

## Run training
```
make train CONFIG_FILE=default.yaml
make train CONFIG_FILE=test.yaml
```