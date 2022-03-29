# -*- coding: utf-8 -*-
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    """
    index handler
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def get(self):
        self.post()

    def post(self):
        self.render("../static/index.html")