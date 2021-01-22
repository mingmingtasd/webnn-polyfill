// Use of this source code is governed by an Apache 2.0 license
// that can be found in the LICENSE file.

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const winston = require('winston');

/**
 * BuilderConf class.
 */
class BuilderConf {
  /**
   * @param {backend} backend Target backend.
   * @param {string} conf Configuration file for build.
   */
  constructor(backend, conf) {
    this.conf_ = conf;
    this.backend_ = backend;

    // Get list via "gn help target_os <out_dir>", current support
    // linux|win
    this.targetOs_ = undefined;

    // Get list via "gn help target_cpu <out_dir>", current support
    // x86|x64|arm|arm64
    this.targetCpu_ = undefined;

    // gn-args
    this.gnArgs_ = {
      isClang: false,
      isComponent: false,
      isDebug: false,
    };

    // True indicate remove out dir before build
    this.cleanBuild_ = true;

    // logging
    this.logFile_ = undefined;
    this.logLevel_ = undefined;
    this.logger_ = undefined;
  }

  /**
   * Init following BuilderConf members from parsing this.conf_:
   *  this.targetOs_
   *  this.targetCpu_
   *  this.gnArgs_
   *  this.cleanBuild_
   *  this.logFile_
   *  this.logLevel_
   *  this.logger_
   */
  init() {
    fs.accessSync(this.conf_);
    const config = JSON.parse(fs.readFileSync(this.conf_, 'utf8'));

    /* jshint ignore:start */
    this.targetOs_ = config['target-os'];
    this.targetCpu_ = config['target-cpu'];
    this.targetOs_ = this.targetOs_ || this.getHostOs();
    this.targetCpu_ = this.targetCpu_ || this.getHostCpu();

    this.gnArgs_.isClang = config['gnArgs']['is-clang'];
    this.gnArgs_.isComponent = config['gnArgs']['is-component'];
    this.gnArgs_.isDebug = config['gnArgs']['is-debug'];

    this.cleanBuild_ = config['clean-build'];

    // Handel logger
    this.logLevel_ = config['logging']['level'] || 'info';
    this.today_ = new Date().toISOString().substring(0, 10);
    this.logFile_ = config['logging']['file'] ||
        path.join(os.tmpdir(),
            'webnn_' + this.targetOs_ + '_' + this.targetCpu_ + '_' +
            this.backend_ + '_' + this.today_ + '.log');
    /* jshint ignore:end */

    this.logger_ = winston.createLogger({
      level: this.logLevel_,
      format: winston.format.simple(),
      transports: [
        new winston.transports.Console({
          colorize: true,
        }),
        new winston.transports.File({
          filename: this.logFile_,
        }),
      ],
    });

    // create logfile is does not exist
    fs.writeFileSync(this.logFile_, '', {flag: 'w+'});
    this.logger_.debug('Config settings:');
    this.logger_.debug(`  backend: ${this.backend_}`);
    this.logger_.debug(`  target OS: ${this.targetOs_}`);
    this.logger_.debug(`  target CPU: ${this.targetCpu_}`);
    this.logger_.debug(`  log level: ${this.logLevel_}`);
    this.logger_.debug(`  log file: ${this.logFile_}`);
  }

  /**
   * @return {string} target backend.
   */
  get backend() {
    return this.backend_;
  }

  /**
   * @return {string} target OS.
   */
  get targetOs() {
    return this.targetOs_;
  }

  /**
   * @return {string} target CPU.
   */
  get targetCpu() {
    return this.targetCpu_;
  }

  /**
   * @return {string} arguments to run 'gn gen'.
   */
  get gnArgs() {
    let args = 'target_os=\"' + this.targetOs + '\"';
    args += ' target_cpu=\"' + this.targetCpu + '\"';
    args += ' is_debug=' + (this.gnArgs_.isDebug).toString();
    args += ' is_component_build=' + (this.gnArgs_.isComponent).toString();
    args += ' is_clang=' + (this.gnArgs_.isClang).toString();
    args += ' dawn_enable_' + this.backend_ + '=true';
    return args;
  }

  /**
   * @return {boolean} of clean build.
   */
  get cleanBuild() {
    return this.cleanBuild_;
  }

  /**
   * @return {string} of today.
   */
  get today() {
    return this.today_;
  }

  /**
   * @return {object} logger.
   */
  get logger() {
    return this.logger_;
  }

  /**
   * @return {string} log file.
   */
  get logFile() {
    return this.logFile_;
  }

  /**
   * @return {string} log level.
   */
  get logLevel() {
    return this.logLevel_;
  }

  /**
   * Get hosted OS string.
   * @return {string} hosted OS.
   */
  getHostOs() {
    const hostOs = os.platform();
    switch (hostOs) {
      case 'linux':
        return 'linux';
      case 'win32':
        return 'win';
      case 'aix':
      case 'freebsd':
      case 'openbsd':
      case 'sunos':
        return 'linux';
    }
  }

  /**
   * Get hosted CPU string.
   * @return {string} hosted CPU.
   */
  getHostCpu() {
    let hostCpu = os.arch();
    switch (hostCpu) {
      case 'arm':
      case 'arm64':
      case 'mipsel':
      case 'x64':
        break;
      case 'ia32':
        hostCpu = 'x86';
        break;
      case 'mips':
      case 'ppc':
      case 'ppc64':
      case 's390':
      case 's390x':
      case 'x32':
        this.logger_.error(`Unsuppurted arch: ${hostCpu}`);
    }

    return hostCpu;
  }
}

module.exports = {
  BuilderConf,
};
