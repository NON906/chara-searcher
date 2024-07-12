# -*- coding: utf-8 -*-

from modules import script_callbacks

from standalone_ui import main_ui

def on_ui_tabs():
    block_interface = main_ui('sd-webui')
    block_interface.queue()
    return [(block_interface, 'chara-searcher', 'chara-searcher_interface')]

script_callbacks.on_ui_tabs(on_ui_tabs)