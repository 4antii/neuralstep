let wavegan = {};

////////////// Config

wavegan.cfg = {
  net: {
    ckptDir: "ckpts",
    ppFilt: true,
    zDim: 100,
    cherries: [5, 2, 0, 62, 55, 12, 56, 21],
  },
  audio: {
    gainDefault: 0.5,
    reverbDefault: 0.25,
    reverbLen: 2,
    reverbDecay: 10,
  },
};

let cfg = wavegan.cfg;

////////////// Net

// Network state
var net = {
  vars: null,
  ready: false,
};

// Hardware state
var hw = {
  math: null,
  ready: false,
};

// Initialize hardware (uses WebGL if possible)
var initHw = function (graph) {
  // TODO: update this
  try {
    new dl.NDArrayMathGPU();
    console.log("WebGL supported");
  } catch (err) {
    new dl.NDArrayMathCPU();
    console.log("WebGL not supported");
  }
  hw.math = dl.ENV.math;
  hw.ready = true;
  console.log("Hardware ready");
};

// Initialize network and hardware
var initVars = function () {
  var varLoader = new dl.CheckpointLoader(cfg.net.ckptDir);
  
  varLoader.getAllVariables().then(function (vars) {
    net.vars = vars;
    net.ready = true;
    console.log(vars);
    console.log("Variables loaded");
  });
};

// Exports
wavegan.net = {};

wavegan.net.isReady = function () {
  return net.ready && hw.ready;
};

wavegan.net.getCherries = function () {
  if (!wavegan.net.isReady()) {
    throw "Hardware not ready";
  }
  if ("cherries" in net.vars) {
    var cherries = net.vars["cherries"];
    var _zs = [];
    for (var i = 0; i < cherries.shape[0]; ++i) {
      var _z = new Float32Array(cfg.net.zDim);
      for (var j = 0; j < cfg.net.zDim; ++j) {
        _z[j] = cherries.get(i, j);
      }
      _zs.push(_z);
    }
    return _zs;
  } else {
    return null;
  }
};

wavegan.net.eval = function (_z) {
  if (!wavegan.net.isReady()) {
    throw "Hardware not ready";
  }
  for (var i = 0; i < _z.length; ++i) {
    if (_z[i].length !== cfg.net.zDim) {
      throw "Input shape incorrect";
    }
  }

  var m = hw.math;

  // Reshape input to 2D array
  var b = _z.length;
  var _z_flat = new Float32Array(b * cfg.net.zDim);
  for (var i = 0; i < b; ++i) {
    for (var j = 0; j < cfg.net.zDim; ++j) {
      _z_flat[i * cfg.net.zDim + j] = _z[i][j];
    }
  }
  var x = dl.Array2D.new([b, cfg.net.zDim], _z_flat);

  // Project to [b, 1, 16, 1024]
  x = m.matMul(x, net.vars["G/z_project/dense/kernel"]);
  x = m.add(x, net.vars["G/z_project/dense/bias"]);
  x = m.relu(x);
  x = x.reshape([b, 1, 16, 1024]);

  // Conv 0 to [b, 1, 64, 512]
  x = m.conv2dTranspose(
    x,
    net.vars["G/upconv_0/conv2d_transpose/kernel"],
    [b, 1, 64, 512],
    [1, 4],
    "same"
  );
  x = m.add(x, net.vars["G/upconv_0/conv2d_transpose/bias"]);
  x = m.relu(x);

  // Conv 1 to [b, 1, 256, 256]
  x = m.conv2dTranspose(
    x,
    net.vars["G/upconv_1/conv2d_transpose/kernel"],
    [b, 1, 256, 256],
    [1, 4],
    "same"
  );
  x = m.add(x, net.vars["G/upconv_1/conv2d_transpose/bias"]);
  x = m.relu(x);

  // Conv 2 to [b, 1, 1024, 128]
  x = m.conv2dTranspose(
    x,
    net.vars["G/upconv_2/conv2d_transpose/kernel"],
    [b, 1, 1024, 128],
    [1, 4],
    "same"
  );
  x = m.add(x, net.vars["G/upconv_2/conv2d_transpose/bias"]);
  x = m.relu(x);

  // Conv 3 to [b, 1, 4096, 64]
  x = m.conv2dTranspose(
    x,
    net.vars["G/upconv_3/conv2d_transpose/kernel"],
    [b, 1, 4096, 64],
    [1, 4],
    "same"
  );
  x = m.add(x, net.vars["G/upconv_3/conv2d_transpose/bias"]);
  x = m.relu(x);

  // Conv 4 to [b, 1, 16384, 1]
  x = m.conv2dTranspose(
    x,
    net.vars["G/upconv_4/conv2d_transpose/kernel"],
    [b, 1, 16384, 1],
    [1, 4],
    "same"
  );
  x = m.add(x, net.vars["G/upconv_4/conv2d_transpose/bias"]);
  x = m.tanh(x);

  // Post processing filter
  x = m.reshape(x, [b, 16384, 1]);
  if (cfg.net.ppFilt) {
    x = m.conv1d(x, net.vars["G/pp_filt/conv1d/kernel"], null, 1, "same");
  }

  // Create Float32Arrays with result
  let wavs = [];
  for (var i = 0; i < b; ++i) {
    var wav = new Float32Array(16384);
    for (var j = 0; j < 16384; ++j) {
      wav[j] = x.get(i, j, 0);
    }
    wavs.push(wav);
  }

  return wavs;
};

// Run immediately
initVars();
initHw();

/*

////////////// SaveWav

var Wav = function(opt_params){
    this._sampleRate = opt_params && opt_params.sampleRate ? opt_params.sampleRate : 44100;
    this._channels = opt_params && opt_params.channels ? opt_params.channels : 2;  
    this._eof = true;
    this._bufferNeedle = 0;
    this._buffer;
};

Wav.prototype.setBuffer = function(buffer){
    this._buffer = this.getWavInt16Array(buffer);
    this._bufferNeedle = 0;
    this._internalBuffer = '';
    this._hasOutputHeader = false;
    this._eof = false;
};

Wav.prototype.getBuffer = function(len){
    var rt;
    if( this._bufferNeedle + len >= this._buffer.length ){
        rt = new Int16Array(this._buffer.length - this._bufferNeedle);
        this._eof = true;
    }
    else {
        rt = new Int16Array(len);
    }
    
    for(var i=0; i<rt.length; i++){
        rt[i] = this._buffer[i+this._bufferNeedle];
    }
    this._bufferNeedle += rt.length;
    
    return  rt.buffer;
};

Wav.prototype.eof = function(){
    return this._eof;
};

Wav.prototype.getWavInt16Array = function(buffer){
    var intBuffer = new Int16Array(buffer.length + 23), tmp;
    
    intBuffer[0] = 0x4952; // "RI"
    intBuffer[1] = 0x4646; // "FF"
    
    intBuffer[2] = (2*buffer.length + 15) & 0x0000ffff; // RIFF size
    intBuffer[3] = ((2*buffer.length + 15) & 0xffff0000) >> 16; // RIFF size
    
    intBuffer[4] = 0x4157; // "WA"
    intBuffer[5] = 0x4556; // "VE"
        
    intBuffer[6] = 0x6d66; // "fm"
    intBuffer[7] = 0x2074; // "t "
        
    intBuffer[8] = 0x0012; // fmt chunksize: 18
    intBuffer[9] = 0x0000; //
        
    intBuffer[10] = 0x0001; // format tag : 1 
    intBuffer[11] = this._channels; // channels: 2
    
    intBuffer[12] = this._sampleRate & 0x0000ffff; // sample per sec
    intBuffer[13] = (this._sampleRate & 0xffff0000) >> 16; // sample per sec
    
    intBuffer[14] = (2*this._channels*this._sampleRate) & 0x0000ffff; // byte per sec
    intBuffer[15] = ((2*this._channels*this._sampleRate) & 0xffff0000) >> 16; // byte per sec
    
    intBuffer[16] = 0x0004; // block align
    intBuffer[17] = 0x0010; // bit per sample
    intBuffer[18] = 0x0000; // cb size
    intBuffer[19] = 0x6164; // "da"
    intBuffer[20] = 0x6174; // "ta"
    intBuffer[21] = (2*buffer.length) & 0x0000ffff; // data size[byte]
    intBuffer[22] = ((2*buffer.length) & 0xffff0000) >> 16; // data size[byte]    

    for (var i = 0; i < buffer.length; i++) {
        tmp = buffer[i];
        if (tmp >= 1) {
            intBuffer[i+23] = (1 << 15) - 1;
        }
        else if (tmp <= -1) {
            intBuffer[i+23] = -(1 << 15);
        }
        else {
            intBuffer[i+23] = Math.round(tmp * (1 << 15));
        }
    }
    
    return intBuffer;
};

wavegan.savewav = {};
wavegan.savewav.randomFilename = function () {
    return Math.random().toString(36).substring(7) + '.wav';
};

wavegan.savewav.saveWav = function (fn, buffer) {
    var wav = new Wav({sampleRate: 16000, channels: 1});
    wav.setBuffer(buffer);

    // Create file
    var srclist = [];
    while (!wav.eof()) {
        srclist.push(wav.getBuffer(1024));
    }
    var b = new Blob(srclist, {type:'audio/wav'});

    // Download
    var URLObject = window.webkitURL || window.URL;
    var url = URLObject.createObjectURL(b);
    var a = document.createElement('a');
    a.style = 'display:none';
    a.href = url;
    a.download = fn;
    a.click();
    URLObject.revokeObjectURL(url);
};
*/

export default {}

