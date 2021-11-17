import subprocess
import os


def run():
    iters = 3
    days =  [7,14,30, 90, 365] #7, 14, 21, 30, 90, 365]
    playersOfEach = [1, 2, 3, 4] 
    groups = set(range(1,10)) 
    groups.remove(8)
    groups.remove(6)
    groups.remove(3)
    path = 'configPath'
    for d in days:
        for j in playersOfEach:
            with open(path, "w") as f:
                # write config
                f.write("group,counts\nrand,0\n")
                for i in groups: 
                    t = j
                    if i == max(groups) and j * len(groups) % 2 == 1:
                        t -= 1
                    f.write(f"g{i},{t}\n")
            for i in range(iters):
                ans = run_program(d, j, i, path)

def run_program(days, j, iterNumber, configPath):
    runString =f"python3.9 main.py --d {days} --config_path {configPath} -p_from_config -remove_round_logging -save_results --run_id run-{days}-{j}-{iterNumber} -restrict_time"
    subprocess.run(runString.split(' '))

if __name__ == "__main__":
    run()
    
