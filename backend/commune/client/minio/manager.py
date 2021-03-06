from minio import Minio
import os
import pickle
import io
import json
from commune.ray.actor import ActorBase
from .utils import get_json, put_json, get_pickle, put_pickle

class MinioManager(ActorBase, Minio):
    default_cfg_path = f"{os.environ['PWD']}/commune/config/client/block/minio.yaml"
    
    def __init__(self, cfg):
        Minio.__init__(self, **cfg['client_kwargs'])

        self.client_kwargs = cfg['client_kwargs']
        self.client = Minio(**self.client_kwargs)

    def bucket_exists(self, bucket):
        return self.client.bucket_exists(bucket)

    
    def get_minio_bucket(self,bucket_name,
                         refresh_bucket=False,
                         verbose=True):
        if  self.bucket_exists(bucket_name):
            if refresh_bucket:
                if verbose:
                    print(f'Re-Creating {bucket_name}')
                # delete all of the content, remove bucket, recreate bucket
                self.delete_folder(bucket_name, '')
                self.client.remove_bucket(bucket_name)
                self.cleint.make_bucket(bucket_name)
            else:
                if verbose:
                    print(f'{bucket_name} Already Exists')
        else:
            self.client.make_bucket(bucket_name)
            if verbose:
                print(f"Successfully Created {bucket_name}")


    def list_objects(self,
                     bucket_name: str,
                     prefix: str,
                     recursive=False):
       return [fo.object_name for fo in list(self.client.list_objects(bucket_name=bucket_name,
                                                                                  prefix=prefix,
                                                                                  recursive=recursive))]

    def delete_folder(self, bucket_name, folder_name):

        for object_name in self.list_objects(bucket_name=bucket_name,prefix=folder_name,recursive=True):
            self.delete_object(bucket_name, object_name)

    def delete_object(self, bucket_name, object_name):
        self.client.remove_object(bucket_name, object_name)

    @staticmethod
    def get_type( path):

        filetype =  path.split('.')[-1]

        if filetype in ['pkl', 'pickle']:
            return 'pickle'
        elif filetype in ['json']:
            return 'json'
        elif filetype in ['onnx']:
            return 'onnx'
        elif filetype in ['torch', 'pth', 'torch.pth', 'model']:
            return 'torch.model'
        else:
            raise Exception('Bruhhhhhh, the file path has to be valid dawg')
    def write(self, bucket_name=None,
                    object_name=None,
                    path=None,
                    data=None,
                    type='pickle'):


        # if type == None:
        #     type = self.get_type(path)

        if path:
            bucket_name, object_name = self.resolve_path(path)



        if type == 'pickle':
            self.put_pickle(bucket_name=bucket_name,
                    object_name=object_name,
                    data=data)
        if type == 'json':
            self.put_json(bucket_name=bucket_name,
                    object_name=object_name,
                    data=data)

        
    def load(self,bucket_name=None,
                    object_name=None,
                    path=None,
                    type='pickle',
                    verbose= False):

                

        if path:
            bucket_name, object_name = self.resolve_path(uri)

        
        if self.has_object(bucket_name=bucket_name,
                           object_name=object_name,
                           ):
            if type == 'pickle':
                return self.get_pickle(bucket_name=bucket_name,
                                  object_name=object_name)
            if type == 'json':
                return self.get_json(bucket_name=bucket_name,
                                object_name=object_name)
        else:
            if verbose:
                print(f'MINIO MANAGER WARNING: obj: {object_name} in bucket: {bucket_name} not found')


    def has_object(self,
                   bucket_name=None,
                   object_name=None,
                   path=None):

        if path:
            bucket_name, object_name = self.resolve_path(uri)

        prefix = '/'.join(object_name.split('/')[:-1])+'/'

        obj_list = [f.object_name for f in 
                        self.client.list_objects(bucket_name,
                          prefix=prefix,
                          recursive=True)
                    if object_name == f.object_name
                    ]

        return bool(obj_list)


    @staticmethod
    def resolve_path(path):
        path = path.split('//')[1]
        path_list = path.split('/')
        bucket_name = path_list[0]
        object_name = '/'.join(path_list[1:])
        return bucket_name, object_name


    def put_json(self, bucket_name, object_name, data):
        """
        jsonify a dict and write it as object to the bucket
        """
        # prepare data and corresponding data stream
        data = json.dumps(data).encode("utf-8")
        data_stream = io.BytesIO(data)
        data_stream.seek(0)

        # put data as object into the bucket
        self.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=data_stream, length=len(data),
            content_type="application/json"
        )
    def put_pickle(self, bucket_name, object_name, data): 
        bytes_file = pickle.dumps(data)
        self.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(bytes_file),
            length=len(bytes_file))


    def get_pickle(self, bucket_name, object_name):
        return pickle.loads(
                        self.client.get_object(
                            bucket_name=bucket_name,
                            object_name=object_name).read()
                    )

    def get_json(self, bucket_name, object_name):
        """
        get stored json object from the bucket
        """
        data = self.client.get_object(bucket_name, object_name)
        return json.load(io.BytesIO(data.data))




    def put_object(self,
                    bucket_name=None, 
                    object_name=None,
                    path=None,
                    data=None,
                    **kwargs ):
        if path:
            bucket_name, object_name = self.resolve_path(path)
        return self.client.put_object(bucket_name=bucket_name, object_name=object_name, data=data, **kwargs)


    def get_object(self,
                    bucket_name=None, 
                    object_name=None,
                    path=None,**kwargs ):
        if path:
            bucket_name, object_name = self.resolve_path(path)
        return self.client.get_object( bucket_name=bucket_name, 
                                        object_name=object_name
                                    ,**kwargs)



    def __del__(self):
        try:
            self.client.__del__()
        except Exception as e:
            self.client.__del__()