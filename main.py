# -*- coding: utf-8 -*-
import argparse
import logging

import tornado
import tornado.ioloop
import tornado.web
import tornado.httpclient
import tornado.httpserver

from lib import base


def run_server(port):
    """
    start YX SERVER
    :return: None
    """

    try:
        app = tornado.web.Application(handlers=[
            (r'/', base.MainHandler),
        ], static_path="static")
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.bind(port, address='0.0.0.0')
        http_server.start(1)
        tornado.ioloop.IOLoop.instance().start()
    except Exception as e:
        logging.error("YX server start failed")
        logging.error(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=False, help="server port", type=int, default=80)

    args = parser.parse_args()
    server_port = args.port
    print(server_port)
    run_server(server_port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
