from commune.process.base import BaseProcess

class GradioAPI(BaseProcess):
    default_cfg_path =  'gradio.api.module'

    def add_port(self, port):
        current = request.json
        visable.append(current)
        return jsonify({"executed" : True})

    def rm_port(self):
        current = request.json
        print(current)
        visable.remove(current)
        return jsonify({"executed" : True,
                        "ports" : current['port']})

    def ls():
        return jsonify(visable)


