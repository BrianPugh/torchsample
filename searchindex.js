Search.setIndex({"docnames": ["Installation", "Overview", "align_corners", "generating_coordinates", "genindex", "index", "models", "nobatch", "positional_encoding", "py-modindex", "sampling", "source/changelog", "source/packages/modules", "source/packages/torchsample", "source/packages/torchsample.coord", "source/packages/torchsample.default", "source/packages/torchsample.encoding", "source/packages/torchsample.encoding.functional", "source/packages/torchsample.encoding.oop", "source/packages/torchsample.models", "source/readme"], "filenames": ["Installation.rst", "Overview.rst", "align_corners.rst", "generating_coordinates.rst", "genindex.rst", "index.rst", "models.rst", "nobatch.rst", "positional_encoding.rst", "py-modindex.rst", "sampling.rst", "source/changelog.rst", "source/packages/modules.rst", "source/packages/torchsample.rst", "source/packages/torchsample.coord.rst", "source/packages/torchsample.default.rst", "source/packages/torchsample.encoding.rst", "source/packages/torchsample.encoding.functional.rst", "source/packages/torchsample.encoding.oop.rst", "source/packages/torchsample.models.rst", "source/readme.rst"], "titles": ["Installation", "Overview", "About Align Corners", "Generating Coordinates", "Index", "Introduction", "Models", "No Batch", "Positional Encoding", "Module Index", "Sampling", "Changelog", "API Reference", "torchsample package", "torchsample.coord module", "torchsample.default module", "torchsample.encoding package", "torchsample.encoding.functional module", "torchsample.encoding.oop module", "torchsample.models module", "Introduction"], "terms": {"torchsampl": [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 20], "requir": [0, 8, 20], "python": [0, 20], "3": [0, 1, 3, 6, 7, 10, 14, 17, 20], "8": [0, 20], "can": [0, 3, 5, 6, 10, 14, 20], "from": [0, 2, 14, 17, 20], "pypi": 0, "via": [0, 17, 20], "m": [0, 2], "pip": [0, 20], "To": [0, 3], "get": [0, 1], "latest": 0, "unreleas": 0, "updat": 0, "directli": [0, 8, 14], "github": [0, 14, 20], "git": [0, 20], "http": [0, 14, 20], "com": [0, 14, 20], "brianpugh": [0, 20], "us": [1, 2, 3, 6, 7, 10, 14], "coordin": [1, 2, 5, 6, 8, 10, 14, 17, 20], "pytorch": [1, 2, 5, 14, 20], "i": [1, 3, 5, 6, 10, 11, 14, 17, 20], "mismash": 1, "between": [1, 3, 6], "normal": [1, 2, 3, 14, 17, 20], "unnorm": [1, 3, 14], "matrix": [1, 10], "indic": 1, "call": [1, 18, 20], "meshgrid": 1, "grid_sampl": 1, "permut": [1, 20], "reshap": 1, "tensor": [1, 3, 7, 8, 14, 17, 19, 20], "aim": 1, "make": [1, 5, 20], "veri": 1, "simpl": [1, 3, 5, 6, 20], "gener": [1, 2, 5, 10, 14, 17], "sampl": [1, 2, 5, 6, 7, 14, 20], "neural": [1, 5, 8, 17, 20], "network": [1, 5, 6, 8, 10, 20], "them": [1, 10, 18], "For": [1, 3, 6, 7], "exampl": [1, 2, 3, 6, 7, 17], "we": [1, 2, 6, 10], "want": [1, 3, 6, 10, 20], "all": [1, 2, 3, 6, 8, 11, 14, 18, 19], "2d": [1, 14], "imag": [1, 2, 3, 5, 7, 10, 14, 17, 20], "queri": [1, 2, 3, 10, 20], "would": [1, 2, 3, 14], "have": [1, 3, 6, 10, 14, 19], "perform": [1, 8, 18], "follow": [1, 7], "import": [1, 17, 20], "torch": [1, 3, 6, 7, 10, 14, 17, 19, 20], "nn": 1, "function": [1, 3, 5, 6, 7, 8, 10, 13, 14, 16, 18, 19, 20], "f": 1, "rand": [1, 3, 7, 8, 10, 14, 17, 20], "1": [1, 2, 3, 5, 7, 14, 17, 20], "480": [1, 3, 7, 10, 17], "640": [1, 3, 7, 10, 17], "unnormalized_coords_x": 1, "unnormalized_coords_i": 1, "arang": 1, "shape": [1, 3, 6, 19, 20], "2": [1, 2, 3, 6, 7, 8, 10, 14, 17, 19, 20], "index": [1, 5, 14], "xy": 1, "These": 1, "ar": [1, 2, 3, 6, 10, 11, 14, 17, 20], "each": [1, 19], "thi": [1, 2, 7, 10, 11, 14, 18, 20], "align_corn": [1, 14, 17, 20], "fals": [1, 10, 14, 17, 20], "normalized_coords_x": 1, "normalized_coords_i": 1, "normalized_coord": 1, "stack": 1, "none": [1, 7, 14, 19], "add": [1, 7], "singleton": [1, 7], "batch": [1, 5, 10, 14, 20], "dimens": [1, 6, 7, 10, 14, 19], "mode": [1, 2, 7, 14], "nearest": [1, 7, 14, 17], "assert": [1, 3], "That": 1, "": [1, 2], "quit": 1, "lot": 1, "work": 1, "dure": [1, 3, 14, 20], "easi": 1, "accident": 1, "swap": 1, "x": [1, 2, 3, 6, 14, 17, 19, 20], "y": [1, 3, 6, 14, 17, 20], "row": [1, 2], "col": 1, "mesh": 1, "creation": 1, "improperli": 1, "wrong": 1, "order": [1, 3, 14, 17, 20], "convers": 1, "let": 1, "see": [1, 6, 10, 14, 18], "how": [1, 2, 19], "look": [1, 2, 7], "t": [1, 3, 6, 7, 8, 10, 14, 17, 20], "coord": [1, 2, 3, 7, 8, 10, 12, 13, 17, 18, 20], "full_lik": [1, 3, 14, 20], "feat_last": [1, 10, 14], "code": [1, 2, 5, 20], "much": [1, 20], "more": [1, 5, 7, 14, 20], "ters": [1, 7], "readabl": [1, 7], "less": 1, "like": [1, 5, 6, 7, 10, 20], "contain": [1, 8], "bug": [1, 2, 20], "allow": 1, "develop": [1, 5, 20], "instead": [1, 17, 18], "focu": [1, 5, 20], "actual": [1, 2], "architectur": 1, "rather": 1, "than": [1, 14], "caught": 1, "up": [1, 2, 14], "machineri": 1, "serv": 2, "brief": 2, "complet": [2, 20], "writeup": 2, "explain": 2, "impact": 2, "when": [2, 7, 8, 14, 20], "should": [2, 18, 19, 20], "set": 2, "unless": [2, 3], "otherwis": [2, 3], "specifi": [2, 6, 19], "refer": [2, 5], "tl": 2, "dr": 2, "proper": 2, "better": 2, "There": 2, "an": [2, 3, 6, 10, 14, 19], "excel": 2, "hollanc": 2, "variou": 2, "framework": 2, "In": [2, 8, 17], "summari": 2, "tensorflowv1": 2, "implement": [2, 20], "bilinear": 2, "resiz": 2, "incorrect": 2, "doubl": 2, "resolut": [2, 3, 14, 17], "result": [2, 5, 14, 20], "shift": 2, "one": [2, 18], "pixel": [2, 3, 14, 17], "left": 2, "also": [2, 10], "repeat": 2, "final": [2, 19], "column": 2, "so": [2, 3, 5, 10, 17, 20], "tensorflow": 2, "attempt": 2, "fix": [2, 11], "backward": 2, "compat": 2, "wai": 2, "caller": [2, 3], "true": [2, 14], "unfortun": 2, "The": [2, 6, 10, 11, 14], "distanc": 2, "success": 2, "depend": 2, "size": [2, 7, 14, 17, 20], "intuit": [2, 20], "semi": 2, "good": 2, "differ": 2, "bilinearli": 2, "link": 2, "articl": 2, "context": 2, "go": [2, 3], "present": 2, "bring": 2, "togeth": 2, "consid": 2, "wide": 2, "tall": 2, "ha": [2, 5, 20], "intens": 2, "100": 2, "right": 2, "200": 2, "point": [2, 14, 20], "center": [2, 17], "0": [2, 3, 5, 7, 14, 17, 20], "5": [2, 14, 17], "norm": 2, "system": 2, "start": 2, "side": [2, 14], "most": 2, "end": [2, 3, 14], "padding_mod": 2, "argument": [2, 3, 8, 20], "By": [2, 6], "default": [2, 6, 10, 12, 13, 14, 19, 20], "zero": 2, "mean": 2, "around": [2, 3], "top": 2, "due": [2, 14], "surround": 2, "pad": 2, "4": 2, "e": [2, 14, 17, 20], "25": 2, "probabl": [2, 14, 20], "activ": [2, 6, 19], "research": 2, "topic": 2, "chang": [2, 5, 11, 20], "same": [2, 6, 14, 20], "neighbor": [2, 17], "valid": 2, "With": 2, "full": [2, 3, 14], "featuremap": [2, 5, 10, 14, 20], "expect": [2, 7], "keep": [3, 11], "uniform": [3, 14], "consist": [3, 20], "api": [3, 5, 20], "everyth": [3, 20], "oper": [3, 7, 8], "off": 3, "rang": [3, 14, 17], "explicitli": 3, "state": 3, "similarli": 3, "alwai": [3, 20], "If": [3, 10, 14], "space": 3, "given": [3, 20], "4d": 3, "typic": 3, "express": 3, "b": [3, 14, 20], "c": [3, 14, 20], "h": [3, 14, 20], "w": [3, 14, 20], "need": [3, 18], "sinc": [3, 7, 18], "associ": 3, "16": [3, 8, 10], "4096": [3, 7, 8, 10, 17, 20], "devic": [3, 14], "cuda": 3, "fall": 3, "exactli": [3, 14], "you": [3, 5, 10, 20], "randint": [3, 7, 14, 17], "despit": 3, "name": 3, "return": [3, 6, 10, 14, 17, 19], "still": 3, "similar": [3, 14], "numpi": 3, "offer": 3, "conveni": 3, "_like": 3, "accept": 3, "5d": 3, "doesn": 3, "juggl": 3, "randint_lik": [3, 14], "infer": [3, 14], "time": 3, "its": [3, 7], "common": [3, 5, 6, 8, 20], "entir": 3, "Or": [3, 20], "creat": [3, 7], "match": [3, 14, 20], "back": 3, "forth": 3, "avail": [3, 10, 20], "xrang": 3, "639": 3, "yrang": 3, "479": 3, "unnormalized_coord": 3, "renormalized_coord": 3, "assert_clos": 3, "lightweight": [5, 20], "warn": [5, 20], "yet": [5, 20], "stabl": [5, 20], "subject": [5, 20], "explicit": [5, 20], "becom": [5, 20], "popular": [5, 20], "learn": [5, 20], "continu": [5, 20], "represent": [5, 20], "local": [5, 20], "implicit": [5, 20], "nerf": [5, 8, 17, 20], "repres": [5, 17, 20], "scene": [5, 17, 20], "radianc": [5, 17, 20], "field": [5, 14, 17, 20], "view": [5, 17, 20], "synthesi": [5, 17, 20], "pointrend": [5, 14, 20], "segment": [5, 14, 17, 20], "render": [5, 14, 20], "provid": [5, 6, 10, 20], "tool": [5, 20], "necessari": [5, 20], "thei": [5, 8, 20], "larg": [5, 20], "amount": [5, 20], "error": [5, 20], "prone": [5, 20], "intend": [5, 14, 20], "other": [5, 20], "part": [5, 14, 20], "model": [5, 8, 12, 13, 14], "instal": 5, "overview": 5, "random": [5, 10, 14], "comprehens": [5, 20], "helper": 5, "posit": [5, 17], "encod": [5, 12, 13], "No": 5, "mlp": [5, 10, 14, 19, 20], "about": 5, "align": [5, 14], "corner": [5, 14], "histori": 5, "explan": [5, 20], "packag": [5, 12], "changelog": 5, "version": [5, 20], "modul": [5, 12, 13, 16], "commonli": 6, "featur": [6, 7, 10, 14], "fed": [6, 10], "class": [6, 18, 19], "multilay": 6, "perceptron": [6, 19], "handl": [6, 10, 14, 20], "arbitrari": 6, "input": [6, 10, 14, 19], "last": [6, 10, 14, 20], "predict": [6, 14], "except": [6, 19], "A": [6, 20], "take": [6, 8, 10, 18], "rgb": [6, 7], "256": [6, 17, 20], "relu": [6, 19], "appli": [6, 10, 19], "layer": [6, 10, 19], "output": [6, 10, 14, 19, 20], "sine": 6, "siren": 6, "could": 6, "sin": [6, 17], "sometim": 7, "mai": 7, "prefer": 7, "without": [7, 14], "dataset": 7, "downstream": [7, 10], "dataload": 7, "collat": 7, "data": [7, 14], "support": 7, "usecas": [7, 20], "either": 7, "suppli": [7, 10], "nobatch": 7, "subfunct": 7, "out": 7, "remov": [7, 11], "do": [7, 20], "Not": 7, "bad": 7, "show": 8, "poorli": 8, "reconstruct": 8, "high": [8, 14, 17], "frequenc": 8, "detail": 8, "were": 8, "abl": 8, "achiev": 8, "significantli": 8, "higher": 8, "vector": 8, "sinusoid": 8, "some": [8, 10, 20], "method": 8, "singl": 8, "produc": 8, "new": 8, "transform": 8, "gamma_encoded_coord": 8, "gamma": [8, 10, 17, 18, 20], "40": 8, "after": 10, "voxel": 10, "frequent": 10, "becaus": 10, "place": [10, 14], "linear": [10, 19], "comput": [10, 18], "effici": 10, "standard": 10, "first": [10, 20], "option": 10, "pass": [10, 18, 19, 20], "concaten": [10, 20], "onto": 10, "43": 10, "notabl": 11, "document": 11, "here": [11, 20], "format": 11, "base": [11, 14, 18, 19], "project": 11, "adher": 11, "semant": 11, "categori": 11, "ad": 11, "deprec": 11, "secur": 11, "releas": 11, "date": 11, "yyyi": 11, "mm": 11, "dd": 11, "initi": 11, "subpackag": 12, "submodul": 12, "oop": [13, 16], "feat_first": 14, "sourc": [14, 17, 18, 19], "move": 14, "dim": [14, 17, 20], "dim1": 14, "3d": 14, "fulli": 14, "n_sampl": 14, "paramet": [14, 17, 19], "int": [14, 17], "don": 14, "tupl": [14, 17, 19], "desir": 14, "bool": [14, 18, 19], "thu": 14, "preserv": 14, "valu": [14, 20], "those": 14, "n": 14, "type": [14, 17, 19], "arg": 14, "kwarg": [14, 18], "length": 14, "dtype": 14, "number": [14, 17], "per": [14, 19], "exemplar": 14, "z": 14, "rand_bias": 14, "pred": [14, 20], "k": 14, "beta": 14, "75": 14, "uncertainti": 14, "describ": 14, "train": [14, 18, 19], "section": 14, "d": 14, "float": 14, "multipli": [14, 17], "oversampl": 14, "bias": 14, "portion": 14, "toward": 14, "uncertain": 14, "region": 14, "lower": 14, "distribut": 14, "ignor": [14, 18], "replac": 14, "unlik": 14, "land": 14, "gt": [14, 20], "load": 14, "numer": 14, "precis": 14, "suggest": 14, "tensor_to_s": 14, "threshold": 14, "report": 14, "score": 14, "feature_extractor": [14, 20], "ss_pred": 14, "decod": 14, "index_coord": 14, "norm_coord": 14, "features_sampl": 14, "refin": 14, "ss_pred_refin": 14, "clone": 14, "flatten": 14, "NOT": 14, "rag": 14, "max": 14, "integ": 14, "doe": 14, "clip": 14, "modifi": 14, "blob": 14, "c52290bad18d23d7decd37b581307f8241c0e8c5": 14, "aten": 14, "src": 14, "nativ": [14, 20], "gridsampl": 14, "l26": 14, "10": 17, "co": 17, "convert": 17, "ident": [17, 18], "unmodifi": 17, "nearest_pixel": [17, 18], "rel": 17, "offset": 17, "note": 17, "qualiti": 17, "ultra": 17, "target": 17, "featmap": [17, 20], "15": 17, "20": 17, "pos_enc": [17, 20], "_oopwrapp": 18, "forward": [18, 19], "defin": 18, "everi": 18, "overridden": 18, "subclass": 18, "although": 18, "recip": 18, "within": 18, "instanc": 18, "afterward": 18, "former": 18, "care": 18, "run": 18, "regist": 18, "hook": 18, "while": 18, "latter": 18, "silent": 18, "nearestpixel": 18, "multi": 19, "__init__": 19, "construct": 19, "node": 19, "list": 19, "mani": 19, "must": 19, "least": 19, "long": 19, "callabl": [19, 20], "inplac": 19, "through": 19, "feat_in": 19, "feat_out": 19, "nightli": 20, "main": 20, "scenario": 20, "randomli": 20, "ground": 20, "truth": 20, "where": 20, "feat": 20, "gt_sampl": 20, "form": 20, "feat_sampl": 20, "scheme": 20, "task": 20, "builtin": 20, "properli": 20, "512": 20, "1024": 20, "touch": 20, "whenev": 20, "It": 20, "simpler": 20, "rule": 20, "help": 20, "prevent": 20, "come": 20, "wrapper": 20, "intent": 20, "clear": 20, "try": 20, "mimic": 20, "torchvis": 20, "interfac": 20, "possibl": 20}, "objects": {"": [[13, 0, 0, "-", "torchsample"]], "torchsample": [[14, 0, 0, "-", "coord"], [15, 0, 0, "-", "default"], [16, 0, 0, "-", "encoding"], [19, 0, 0, "-", "models"]], "torchsample.coord": [[14, 1, 1, "", "feat_first"], [14, 1, 1, "", "feat_last"], [14, 1, 1, "", "full"], [14, 1, 1, "", "full_like"], [14, 1, 1, "", "normalize"], [14, 1, 1, "", "rand"], [14, 1, 1, "", "rand_biased"], [14, 1, 1, "", "randint"], [14, 1, 1, "", "randint_like"], [14, 1, 1, "", "tensor_to_size"], [14, 1, 1, "", "uncertain"], [14, 1, 1, "", "unnormalize"]], "torchsample.encoding": [[17, 0, 0, "-", "functional"], [18, 0, 0, "-", "oop"]], "torchsample.encoding.functional": [[17, 1, 1, "", "gamma"], [17, 1, 1, "", "identity"], [17, 1, 1, "", "nearest_pixel"]], "torchsample.encoding.oop": [[18, 2, 1, "", "Gamma"], [18, 2, 1, "", "Identity"], [18, 2, 1, "", "NearestPixel"]], "torchsample.encoding.oop.Gamma": [[18, 3, 1, "", "forward"], [18, 4, 1, "", "training"]], "torchsample.encoding.oop.Identity": [[18, 3, 1, "", "forward"], [18, 4, 1, "", "training"]], "torchsample.encoding.oop.NearestPixel": [[18, 3, 1, "", "forward"], [18, 4, 1, "", "training"]], "torchsample.models": [[19, 2, 1, "", "MLP"]], "torchsample.models.MLP": [[19, 3, 1, "", "__init__"], [19, 3, 1, "", "forward"], [19, 4, 1, "", "training"]]}, "objtypes": {"0": "py:module", "1": "py:function", "2": "py:class", "3": "py:method", "4": "py:attribute"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"], "2": ["py", "class", "Python class"], "3": ["py", "method", "Python method"], "4": ["py", "attribute", "Python attribute"]}, "titleterms": {"instal": [0, 20], "overview": 1, "about": 2, "align": 2, "corner": 2, "histori": 2, "explan": 2, "For": 2, "align_corn": 2, "fals": 2, "what": 2, "border": 2, "valu": 2, "i": 2, "interpol": 2, "between": 2, "That": 2, "sound": 2, "bad": 2, "gener": 3, "coordin": 3, "random": 3, "comprehens": 3, "helper": 3, "index": [4, 9], "introduct": [5, 20], "content": 5, "model": [6, 19, 20], "mlp": 6, "No": 7, "batch": 7, "posit": [8, 10, 20], "encod": [8, 10, 16, 17, 18, 20], "modul": [9, 14, 15, 17, 18, 19], "sampl": 10, "changelog": 11, "version": 11, "0": 11, "1": 11, "api": 12, "refer": 12, "torchsampl": [13, 14, 15, 16, 17, 18, 19], "packag": [13, 16], "subpackag": 13, "submodul": [13, 16], "coord": 14, "default": 15, "function": 17, "oop": 18, "usag": 20, "train": 20, "infer": 20, "design": 20, "decis": 20}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})