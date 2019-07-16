# _*_coding:utf-8_*_
from __future__ import print_function

import json
import os
import re
import socket
import sys
import time


try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

import requests


class StanfordCoreNLP:
    def __init__(self, path_or_host, port=None, memory='4g', lang='en', timeout=1500, quiet=True):
        self.port = port
        self.memory = memory
        self.lang = lang
        self.timeout = timeout
        self.quiet = quiet
        self.url = path_or_host + ':' + str(port)

        # Wait until server starts
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_name = urlparse(self.url).hostname
        while sock.connect_ex((host_name, self.port)):
            time.sleep(1)

    def annotate(self, text, properties=None):
        if sys.version_info.major >= 3:
            text = text.encode('utf-8')

        r = requests.post(self.url, params={'properties': str(properties)}, data=text,
                          headers={'Connection': 'close'})
        return r.text

    # def word_tokenize(self, sentence, span=False):
    #     r_dict = self._request('ssplit,tokenize', sentence)
    #     tokens = [token['originalText'] for s in r_dict['sentences'] for token in s['tokens']]
    #
    #     # Whether return token span
    #     if span:
    #         spans = [(token['characterOffsetBegin'], token['characterOffsetEnd']) for s in r_dict['sentences'] for token
    #                  in s['tokens']]
    #         return tokens, spans
    #     else:
    #         return tokens

    def pos_tag(self, sentence):
        r_dict = self._request('pos', sentence)
        words = []
        tags = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['originalText'])
                tags.append(token['pos'])
        return list(zip(words, tags))

    # def ner(self, sentence):
    #     r_dict = self._request('ner', sentence)
    #     words = []
    #     ner_tags = []
    #     for s in r_dict['sentences']:
    #         for token in s['tokens']:
    #             words.append(token['originalText'])
    #             ner_tags.append(token['ner'])
    #     return list(zip(words, ner_tags))
    #
    # def parse(self, sentence):
    #     r_dict = self._request('pos,parse', sentence)
    #     return [s['parse'] for s in r_dict['sentences']][0]
    #
    def dependency_parse(self, sentence):
        r_dict = self._request('depparse', sentence)
        print(r_dict)
        return [(dep['dep'], dep['governor'], dep['dependent']) for s in r_dict['sentences'] for dep in
                s['basicDependencies']]

    def _request(self, annotators=None, data=None, *args, **kwargs):
        if sys.version_info.major >= 3:
            data = data.encode('utf-8')

        properties = {'annotators': annotators, 'outputFormat': 'json', "tokenize.whitespace": "true"}
        params = {'properties': str(properties), 'pipelineLanguage': self.lang}
        if 'pattern' in kwargs:
            params = {"pattern": kwargs['pattern'], 'properties': str(properties), 'pipelineLanguage': self.lang}

        r = requests.post(self.url, params=params, data=data, headers={'Connection': 'close'})
        r_dict = json.loads(r.text)

        return r_dict
