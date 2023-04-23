"""
    Batch-generate data
"""

import os
import numpy as np
import multiprocessing as mp
from subprocess import call
import time


class DataGen(object):

    def __init__(self, num_processes, flog=None):
        self.num_processes = num_processes
        self.flog = flog
        
        self.todos = []
        self.processes = []
        self.is_running = False
        self.Q = mp.Queue()

    def __len__(self):
        return len(self.todos)

    def add_one_collect_job(self, data_dir, shape_id, category, cnt_id, primact_type, trial_id, state=None):
        if self.is_running:
            print('ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('COLLECT', shape_id, category, cnt_id, primact_type, data_dir, trial_id, np.random.randint(10000000), state)
        self.todos.append(todo)
    
    def add_one_recollect_job(self, src_data_dir, dir1, dir2, recollect_record_name, tar_data_dir, x, y, bias=None):
        if self.is_running:
            print('ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('RECOLLECT', src_data_dir, recollect_record_name, tar_data_dir, np.random.randint(10000000), x, y, dir1, dir2, bias)
        self.todos.append(todo)
    
    def add_one_checkcollect_job(self, src_data_dir, dir1, dir2, recollect_record_name, tar_data_dir, x, y):
        if self.is_running:
            print('ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('CHECKCOLLECT', src_data_dir, recollect_record_name, tar_data_dir, np.random.randint(10000000), x, y, dir1, dir2)
        self.todos.append(todo)
    
    @staticmethod
    def job_func(pid, todos, Q):
        succ_todos = []
        for todo in todos:
            if todo[0] == 'COLLECT':
                if todo[8] is not None:
                    cmd = 'xvfb-run -a python collect_data.py %s %s %d %s --out_dir %s --trial_id %d --random_seed %d --no_gui --state %s' \
                          % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7], todo[8])
                else:
                    cmd = 'xvfb-run -a python collect_data.py %s %s %d %s --out_dir %s --trial_id %d --random_seed %d --no_gui' \
                            % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7])
                folder_name = todo[5]
                # print(f'save to {folder_name}')
                job_name = '%s_%s_%d_%s_%s' % (todo[1], todo[2], todo[3], todo[4], todo[6])
            elif todo[0] == 'RECOLLECT':
                cmd = 'xvfb-run -a python recollect_data.py %s %s %s --random_seed %d --no_gui --x %d --y %d --dir1 %s --dir2 %s --bias %d ' \
                        % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7], todo[8], todo[9])
                folder_name = todo[3]
                if todo[9] is not None:
                    trial_id = todo[2].split('_')[-1]
                    job_name = todo[2].rstrip(trial_id)
                    trial_id = str(int(trial_id) + todo[9])
                    job_name = job_name + trial_id
                    # print(job_name)
                else:
                    job_name = todo[2]
            elif todo[0] == 'CHECKCOLLECT':
                cmd = 'xvfb-run -a python checkcollect_data.py %s %s %s --random_seed %d --no_gui --x %d --y %d --dir1 %s --dir2 %s > /dev/null 2>&1' \
                        % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7], todo[8])
                folder_name = todo[3]
                job_name = todo[2]
            ret = call(cmd, shell=True)
            # print("cmd start")
            if ret == 0:
                # print("cmd succ")
                succ_todos.append(os.path.join(folder_name, job_name))
            if ret == 2:
                succ_todos.append(None)
        Q.put(succ_todos)

    def start_all(self):
        if self.is_running:
            print('ERROR: cannot start all while DataGen is running!')
            exit(1)

        total_todos = len(self)
        if self.num_processes != 0:
            num_todos_per_process = int(np.ceil(total_todos / self.num_processes))
        else:
            num_todos_per_process = 0
        np.random.shuffle(self.todos)
        for i in range(self.num_processes):
            todos = self.todos[i*num_todos_per_process: min(total_todos, (i+1)*num_todos_per_process)]
            p = mp.Process(target=self.job_func, args=(i, todos, self.Q))
            p.start()
            self.processes.append(p)
        
        self.is_running = True

    def join_all(self):
        if not self.is_running:
            print('ERROR: cannot join all while DataGen is idle!')
            exit(1)

        ret = []
        for p in self.processes:
            ret += self.Q.get()

        for p in self.processes:
            p.join()

        self.todos = []
        self.processes = []
        self.Q = mp.Queue()
        self.is_running = False
        return ret


