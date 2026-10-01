"""Switch the catalogue-wide index off (the service falls back to the classics index on its next wake).
  modal run worker/recognise_off.py"""
import modal
app = modal.App("sonic-recognise-off")
vol = modal.Volume.from_name("sonic-recognise", create_if_missing=True)

@app.function(volumes={"/idx": vol}, timeout=120)
def off():
    import os
    p = "/idx/all/ACTIVE"
    if os.path.exists(p):
        os.remove(p); vol.commit(); return "catalogue index switched off: the service uses the classics index from its next wake"
    return "catalogue index was already off"

@app.local_entrypoint()
def main():
    print("::notice title=recognise::" + off.remote())
