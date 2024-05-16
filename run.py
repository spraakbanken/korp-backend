import os
# Patch standard library to cooperate with gevent (https://www.gevent.org/api/gevent.monkey.html)
# Skip patching if run through gunicorn (which does the patching for us)
if "gunicorn" not in os.environ.get("SERVER_SOFTWARE", ""):
    from gevent import monkey
    monkey.patch_all(subprocess=False)  # Patching needs to be done as early as possible, before other imports
else:
    # gunicorn patches everything, and gevent's subprocess module can't be used in
    # native threads other than the main one, so we need to un-patch the subprocess module.
    import subprocess
    from importlib import reload
    reload(subprocess)

import sys

from gevent.pywsgi import WSGIServer

from korp import create_app

if __name__ == "__main__":
    app = create_app()

    if len(sys.argv) == 2 and sys.argv[1] == "dev":
        # Run using Flask (use only for development)
        app.run(debug=True, threaded=True, host=app.config["WSGI_HOST"], port=app.config["WSGI_PORT"])
    else:
        # Run using gevent
        print("Serving using gevent")
        http = WSGIServer((app.config["WSGI_HOST"], app.config["WSGI_PORT"]), app.wsgi_app)
        http.serve_forever()
