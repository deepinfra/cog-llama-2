Setup
=====

* download cog: https://github.com/replicate/cog/releases/
* download a llama.cpp quantization from https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML/tree/main, place in weights/
* tweak predict.py `MODEL` variable to match your weights
* try sample inference with `cog predict -i prompt="What came first, the chicken or the egg?"`
* make sure to put the right image in `cog.yaml` so the container name is ready to push out-of-the box
* once you're ready run `cog build`, test it one more time with

```bash
docker run --gpus all --rm -it -p 5000:5000 IMAGE_NAME
# in another terminal, after it starts
curl -X POST http://127.0.0.1:5000/predictions \
    --data '{"input": {"prompt": "Hello"}}' \
    -H 'Content-Type: application/json' \
    | python -m json.tool
```

* (optional) push image with `docker push IMAGE_NAME`

Notes
=====

* the `EXTRA` flags at the top of `predict.py` are for 70B (and 35B) llama-2 models. For 13B and under, drop `-gqa` and `-eps`, i.e leave `-ngl`.
* the setup function starts the llama server and waits for it to become available
* for predictions we query the HTTP server
* for `cog build --separate-weights` you might need a more recent cog 0.8.4+ with some kinks fixed
