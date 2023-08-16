import paramiko
import os

def retrieve_file(host, username, password, remote_path, local_path):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, username=username, password=password)
    
    ftp_client=ssh_client.open_sftp()
    
    if not os.path.exists(local_path):
        ftp_client.get(remote_path,local_path)
        
    ftp_client.close()
    ssh_client.close()
    
    
    
def retrive_dir(host, username, password, remote_path, local_path, exceptions = None):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, username=username, password=password)
    
    stdin,stdout,stderr = ssh_client.exec_command('ls {}'.format(remote_path))
    directory_names = [x.rstrip() for x in stdout.readlines()]
    retrieved_files = {}
    
    ftp_client=ssh_client.open_sftp()
    
    for directory in directory_names:
        if directory in exceptions:
            continue
        if not os.path.exists('{}/{}'.format(local_path, directory)):
            os.mkdir('{}/{}'.format(local_path, directory))
        stdin,stdout,stderr = ssh_client.exec_command('ls {}/{}'.format(remote_path, directory))
        file_names = [x.rstrip() for x in stdout.readlines()]
        retrieved_files.update({directory : file_names})
        for index, file in enumerate(file_names):
            if file in exceptions:
                retrieved_files[directory].pop(index)
                continue
            if not os.path.exists('{}/{}/{}'.format(local_path, directory, file)):
                ftp_client.get('{}/{}/{}'.format(remote_path, directory, file),'{}/{}/{}'.format(local_path, directory, file))
                    
    ftp_client.close()
    ssh_client.close()
    
    return retrieved_files