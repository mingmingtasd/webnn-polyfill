// Use of this source code is governed by an Apache 2.0 license
// that can be found in the LICENSE file.

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const rimraf = require('rimraf');
const {spawn} = require('child_process');
const {BuilderConf} = require('./builder_conf');

/**
 * Builder class.
 */
class Builder {
  /**
   * @param {string} rootDir webnn-native source code directory.
   */
  constructor(rootDir) {
    this.rootDir_ = rootDir;
    this.outDir_ = path.join(this.rootDir_, 'out', 'Shared');
    this.config_ = undefined;
    this.childResult_ = {};
  }

  /**
   * Init config_ member.
   * @param {object} options An object containing options as key-value pairs.
   */
  initConfig(options) {
    let backend = 'null';
    let configFile = 'build_script/bot_config.json';

    if (options !== undefined) {
      backend = options.backend;
      configFile = options.config;
    }

    if (!path.isAbsolute(configFile)) {
      configFile = path.join(this.rootDir_, configFile);
    }

    this.config_ = new BuilderConf(backend, configFile);
    this.config_.init();
    this.config_.logger.debug('root dir: ' + this.rootDir_);
    this.config_.logger.debug('out dir: ' + this.outDir_);
    this.config_.logger.debug(`config file: ${configFile}`);
  };

  /**
   * Run 'gclient sync' command
   */
  async actionSync() {
    this.config_.logger.info('Action sync');
    await this.childCommand(
      os.platform() == 'win32' ? 'gclient.bat' : 'gclient',
      ['sync'], this.rootDir_);

    if (!this.childResult_.success) {
      this.config_.logger.error('Failed run \'gclient sync\' command.');
      process.exit(1);
    }
  }

  /**
   * Run 'git pull' command
   */
  async actionPull() {
    this.config_.logger.info('Action pull');
    await this.childCommand('git', ['pull', '--rebase'], this.rootDir_);

    if (!this.childResult_.success) {
      this.config_.logger.error('Failed run \'git pull\' command.');
      process.exit(1);
    }
  }

  /**
   * Run 'gn gen' and 'ninja -C' commands
   */
  async actionBuild() {
    this.config_.logger.info('Action build');

    if (this.config_.cleanBuild) {
      if (fs.existsSync(this.outDir_)) {
        rimraf.sync(this.outDir_);
      }
    }

    let genCmd = ['gen', `--args=${this.config_.gnArgs}`, this.outDir_];

    if (this.config_.backend === 'win') {
      genCmd =
        ['gen', `--ide=vs --args=${this.config_.gnArgs}`, this.outDir_];
    }

    await this.childCommand(
      os.platform() == 'win32' ? 'gn.bat' : 'gn', genCmd, this.rootDir_);

    if (!this.childResult_.success) {
      this.config_.logger.error('Failed run \'gn gen\' command.');
      process.exit(1);
    }

    const args = ['-C', this.outDir_];
    await this.childCommand(
        'ninja', args, this.rootDir_);
    if (!this.childResult_.success) {
      process.exit(1);
    }
  }

  /**
   * Run Package command
   */
  async actionPackage() {
    this.config_.logger.info('Action package');
    // TODO
  }

  /**
   * Run upload command
   */
  async actionUpload() {
    this.config_.logger.info('Action upload');
    // TODO
  }

  /**
   * Run upload command
   */
  async actionNotify() {
    this.config_.logger.info('Action notfy');
    // TODO
  }

  /**
   * Run
   */
  async run(cmd, options) {
    this.initConfig(options);

    switch (cmd) {
      case 'sync':
        await this.actionSync();
        break;
      case 'pull':
        await this.actionPull();
        break;
      case 'build':
        await this.actionBuild();
        break;
      case 'package':
        await this.actionPackage();
        break;
      case 'upload':
        await this.actionUpload();
        break;
      case 'notify':
        await this.actionNotify();
        break;
      case 'all':
        await this.actionSync();
        await this.actionPull();
        await this.actionBuild();
        await this.actionPackage();
        await this.actionUpload();
        await this.actionNotify();
        break;
      default:
        this.config_.logger.error(`Unsupported command: ${cmd}`);
        process.exit(1);
    }
  }

  /**
   * Execute command.
   * @param {string} cmd command string.
   * @param {array} args arguments array.
   * @param {string} cwd path string.
   * @param {object} result return value.
   * @return {object} child_process.spawn promise.
   */
  childCommand(cmd, args, cwd, result) {
    return new Promise((resolve, reject) => {
      const cmdFullStr = cmd + ' ' + args.join(' ');
      this.config_.logger.info('Execute command: ' + cmdFullStr);
      const child = spawn(cmd, [...args], {cwd: cwd});

      child.stdout.on('data', (data) => {
        if (result) result.changeset = data.toString();
        this.config_.logger.debug(`${data.toString()}`);
      });

      child.stderr.on('data', (data) => {
        this.config_.logger.error(data.toString());
      });

      child.on('close', (code) => {
        this.childResult_.success = (code === 0);
        resolve(code);
      });
    });
  }
}

module.exports = {
  Builder,
};
