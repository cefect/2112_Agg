'''
Created on Sep. 9, 2022

@author: cefect
'''
from dask.distributed import Client, LocalCluster

if __name__ == "__main__":
    with LocalCluster(n_workers=4,
            processes=True,
            threads_per_worker=1,
            memory_limit='2GB',
            ip='tcp://localhost:9895',
        ) as cluster, Client(cluster) as client:
        print('started dask client w/ dashboard at \n    %s'%client.dashboard_link) #get the link to the dsashbaord
        pass