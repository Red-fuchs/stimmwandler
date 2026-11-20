Third-party code

src/stimmwandler/rvc/ is vendored from RVC-Project/Retrieval-based-Voice-Conversion-WebUI (MIT), with imports adjusted and the RMVPE inference wrapper removed.

MIT License

Copyright (c) 2023 liujing04
Copyright (c) 2023 源文雨
Copyright (c) 2023 Ftps

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


---

src/stimmwandler/knn/ is vendored from bshall/knn-vc (MIT, Copyright (c) 2023 MediaLab, Department of Electrical & Electronic Engineering, Stellenbosch University). Contains:
- wavlm.py, modules.py: WavLM from microsoft/unilm (MIT, Copyright (c) 2021 Microsoft), based on fairseq (MIT).
- hifigan.py: HiFi-GAN generator, based on jik876/hifi-gan (MIT, Copyright (c) 2020 Jungil Kong).

Model weights are NOT redistributed; they are downloaded at first use from the bshall/knn-vc GitHub release v0.1 (WavLM-Large.pt = Microsoft WavLM Large checkpoint, prematch_g_02500000.pt = HiFi-GAN trained by the kNN-VC authors). The kNN-VC repository is MIT; the WavLM repository is MIT; neither states a separate weights license, but WavLM-Large was pre-trained on Libri-Light, GigaSpeech and VoxPopuli, whose terms apply to you. The HiFi-GAN was trained on LibriSpeech (CC BY 4.0). Check these yourself before commercial use.

The MIT permission notice of the upstream repositories is the same text as above with the respective copyright holders.
