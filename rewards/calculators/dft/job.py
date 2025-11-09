import os
import time
from copy import deepcopy
from typing import List
import subprocess

import yaml
import paramiko


SCHEDULER_CMD = {
    'slurm': {
        'submit': ['sbatch', 'INPUT'],
        'state': ['squeue', '--job', 'INPUT']
    }
}


def get_scheduler_cmd(scheduler, task, args, out_str=False):
    cmd = deepcopy(SCHEDULER_CMD[scheduler][task])
    for i in range(len(cmd)):
        if cmd[i] == 'INPUT':
            cmd[i] = args

    if out_str:
        cmd = " ".join(cmd)

    return cmd


class RemoteQueueJob:
    def __init__(
        self,
        hostname: str,
        username: str,
        port: int,
        scheduler: str,
        remote_dir: str,
        script_str: str,
        password: str = None,
        key_path: str = None,
        result_path: str = None,
        forward_file: List[str] = None,
        backward_file: List[str] = None,
    ) -> None:
        self.hostname = hostname
        self.username = username
        self.port = port
        self.scheduler = scheduler
        self.remote_dir = remote_dir
        self.script_str = script_str
        self.password = password
        self.key_path = key_path
        self.result_path = result_path
        self.forward_file = forward_file
        self.backward_file = backward_file
        self.ssh = None
        self.job_id = None

        if result_path is None:
            self.result_path = os.path.join(remote_dir, 'DFTScoreResults')

    def connect(self):
        self.ssh = paramiko.SSHClient()
        # self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connected = False
        while not connected:
            try:
                if self.key_path is not None:
                    self.ssh.connect(self.hostname, username=self.username, port=self.port, key_filename=self.key_path, timeout=15, banner_timeout=15)
                elif self.password is not None:
                    self.ssh.connect(self.hostname, username=self.username, port=self.port, password=self.password, timeout=15, banner_timeout=15)
                else:
                    self.ssh.load_system_host_keys()
                    self.ssh.connect(self.hostname, username=self.username, port=self.port, timeout=15, banner_timeout=15)
                connected = True
            except:
                time.sleep(3)

    def create_dir(self):
        self.connect()
        cmd = f'mkdir -p  {self.remote_dir}'
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        self.ssh.close()

    def file_transfer(self, file_list: List[str]):
        self.connect()
        sftp = self.ssh.open_sftp()

        for file_path in file_list:
            file_name = os.path.basename(file_path)
            remote_path = os.path.join(self.remote_dir, file_name)
            sftp.put(file_path, remote_path)

        sftp.close()
        self.ssh.close()

    def write_submit_job(self):
        self.connect()
        sftp = self.ssh.open_sftp()
        remote_path = os.path.join(self.remote_dir, 'sub.sh')
        with sftp.file(remote_path, "w") as remote_file:
            remote_file.write(self.script_str)
        # sftp.chmod(remote_path, 0o755)
        sftp.close()

        submit_cmd = get_scheduler_cmd(
            self.scheduler,
            'submit',
            remote_path,
            out_str=True,
        )
        submit_cmd = f'cd {self.remote_dir} && ' + submit_cmd
        stdin, stdout, stderr = self.ssh.exec_command(submit_cmd)
        stdout = stdout.read().decode().strip()

        # Get job ID from the output
        # The job ID is usually the last part of sbatch's output
        try:
            self.job_id = stdout.split()[-1]
        except:
            print(stderr.decode())
            raise RuntimeError(f'Submitting a {self.scheduler} job failed')

        return self.job_id

    # def exec_command(self):
    #     self.connect()
    #     for c in self.commands:
    #         stdin, stdout, stderr = self.ssh.exec_command(c)
    #     command_out = stdout.read().decode()
    #     command_err = stderr.read().decode()
    #     assert 'SUBMITTED JOB ID' in command_out, \
    #         'Failed to run dft_score on remote'
    #     self.job_id = command_out.strip().split()[-1]
    #     self.ssh.close()

    def check_status(self):
        assert self.job_id is not None
        self.connect()
        state_cmd = get_scheduler_cmd(
            self.scheduler,
            'state',
            self.job_id,
            out_str=True,
        )
        stdin, stdout, stderr = self.ssh.exec_command(state_cmd)
        stdout = stdout.read().decode()
        self.ssh.close()

        # TODO: Add more job status
        if self.job_id in stdout.strip():
            self.job_status = "RUNNING"
        else:
            self.job_status =  "END"
        return self.job_status

    def wait_job_end(self, check_interval: int = 60):
        while True:
            self.check_status()
            if self.job_status == "END":
                # print(f"Job {job_id} has completed.")
                break
            else:
                # print(f"Job {job_id} is still running.")
                time.sleep(check_interval)

    def read_results(self):
        self.connect()
        stdin, stdout, stderr = self.ssh.exec_command(
            f'cat {self.result_path}'
        )
        results = stdout.read().decode()
        self.ssh.close()

        assert results != '', 'Failed to read results on remote'
        return results.strip()

    def submit_wait_read(self):
        self.create_dir()
        self.file_transfer(self.forward_file)
        self.write_submit_job()
        self.wait_job_end()
        results = self.read_results()

        return results

    @classmethod
    def from_config(
        cls,
        config: dict,
        **kwargs,
    ):
        for k, v in kwargs.items():
            config[k] = v

        remote_dir = os.path.join(config['remote_dir'], config['dir'])

        forward_file = []
        task_cmd = f"dft_score --task {config['task']} --dir {remote_dir}"
        if 'config' in config.keys():
            forward_file.append(config['config'])
            remote_config_path = os.path.join(
                remote_dir,
                os.path.basename(config['config']),
            )
            task_cmd = task_cmd + f' --config {remote_config_path}'

        if 'cif' in config.keys():
            forward_file.append(config['cif'])
            remote_cif_path = os.path.join(
                remote_dir,
                os.path.basename(config['cif']),
            )
            task_cmd = task_cmd + f' --cif {remote_cif_path}'

        if 'smiles' in config.keys():
            task_cmd = task_cmd + ' --smiles ' + config['smiles']

        task_cmd = task_cmd + ' --machine local --scheduler no'
        script_str = config['scheduler_cmd'] + '\n' + task_cmd

        job = cls(
            hostname=config['hostname'],
            username=config['username'],
            port=config['port'],
            scheduler=config['scheduler'],
            remote_dir=remote_dir,
            script_str=script_str,
            forward_file=forward_file,
        )

        return job


class QueueJob:
    """ for local job"""
    def __init__(
        self,
        work_dir: str,
        scheduler: str,
        cmd_str: str = None,
        prev_cmd: List[str] = [],
        task_cmd: List[str] = [],
        result_path: str = None,
        **kwargs,
    ) -> None:
        assert scheduler in SCHEDULER_CMD.keys()
        self.scheduler = scheduler

        self.work_dir = os.path.abspath(work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        if cmd_str is None:
            self.cmd_str = prev_cmd + task_cmd
            assert len(self.cmd_str) > 0, "Provide commands of job!"
            self.cmd_str = "\n".join(self.cmd_str)

        # self.jobname = jobname
        self.result_path = result_path
        self.script_path = None
        self.job_id = None
        self.job_status = None
        if self.result_path is None:
            self.result_path = os.path.join(self.work_dir, 'DFTScoreResults')
        if os.path.exists(self.result_path):
            os.remove(self.result_path)

    def write_script(self, fname='sub.sh'):
        self.script_path = os.path.join(self.work_dir, fname)
        with open(self.script_path, 'w') as f:
            f.write(self.cmd_str)

    def submit(self):
        assert self.script_path is not None
        submit_cmd = get_scheduler_cmd(
            self.scheduler,
            'submit',
            self.script_path,
        )

        # Submit the SLURM job
        initial_dir = os.getcwd()
        os.chdir(self.work_dir)
        result = subprocess.run(
            submit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = result.stdout.decode().strip()
        os.chdir(initial_dir)

        # Get job ID from the output
        # The job ID is usually the last part of sbatch's output
        try:
            self.job_id = output.split()[-1]
        except:
            print(result.stderr.decode())
            raise RuntimeError(f'Submitting a {self.scheduler} job failed')
        # print(f"Job submitted with ID: {job_id}")
        return self.job_id

    def check_status(self):
        assert self.job_id is not None
        state_cmd = get_scheduler_cmd(
            self.scheduler,
            'state',
            self.job_id,
        )

        result = subprocess.run(
            state_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _status = result.stdout.decode()

        # TODO: Add more job status
        if self.job_id in _status.strip():
            self.job_status = "RUNNING"
        else:
            self.job_status =  "END"

        return self.job_status

    def wait_job_end(self, check_interval=5):
        while True:
            self.check_status()
            if self.job_status == "END":
                # print(f"Job {job_id} has completed.")
                break
            else:
                # print(f"Job {job_id} is still running.")
                time.sleep(check_interval)

    def write_submit(self):
        self.write_script()
        job_id = self.submit()

        return job_id

    def submit_wait_read(self):
        self.write_script()
        self.submit()
        self.wait_job_end()
        results = self.read_results()

        return results

    @classmethod
    def from_config(
        cls,
        config: dict,
        **kwargs,
    ):
        for k, v in kwargs.items():
            config[k] = v

        work_dir = os.path.abspath(config['dir'])
        os.makedirs(work_dir, exist_ok=True)
        config_path = os.path.join(work_dir, 'config.yaml')

        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, allow_unicode=True, default_flow_style=False)

        task_cmd = f"dft_score --task {config['task']} --config {config_path}"
        task_cmd = task_cmd + f' --dir {work_dir}'
        task_cmd = task_cmd + ' --machine local --scheduler no'
        cmd_str = config['scheduler_cmd'] + '\n' + task_cmd

        job = cls(
            work_dir=work_dir,
            scheduler=config['scheduler'],
            cmd_str=cmd_str,
        )

        return job
