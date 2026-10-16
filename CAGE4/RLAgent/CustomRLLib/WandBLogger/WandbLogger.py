from ray.tune.logger import Logger
import json

def json_serializable(item):
    try:
        json.dumps(item)
        return True
    except TypeError:
        return False

class WandbLogger(Logger):
    """Logs results by simply printing out everything."""

    '''
    def _init(self):
        # Custom init function.
        print("Initializing wandb...")
        if not hasattr(self, 'wandb_initialized'):
            project_name = self.config.get("project")
            wandb.init(project=project_name, entity="wen-nico")
            self.wandb_initialized = True

    def on_result(self, result: dict):
        if not hasattr(self, 'config_intialized'):
            wandb.config.update(result['config'])
            self.config_intialized = True
        
        serializable_result = {}
        for key, value in result.items():
            if json_serializable(value):
                serializable_result[key] = value
            else:
                serializable_result[key] = str(value)
            
        wandb.log(serializable_result)  

        print(result)

    def close(self):
        # Releases all resources used by this logger.
        print("Closing")
        wandb.finish()

    def flush(self):
        # Flushing all possible disk writes to permanent storage.
        print("Flushing", flush=True)
    '''