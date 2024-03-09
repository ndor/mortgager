# import numpy as np
from pyscript import document


def translate_english(event):
    input_text = document.querySelector("#button1")
    english = input_text.value
    output_div = document.querySelector("#output")
    output_div.innerText = '50% מס'
