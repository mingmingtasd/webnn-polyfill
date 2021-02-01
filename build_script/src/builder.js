// Use of this source code is governed by an Apache 2.0 license
// that can be found in the LICENSE file.

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const rimraf = require('rimraf');
const crypto = require('crypto');
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

    // Get last sucessful changeset
    this.lastSucceedChangesetFile_ = path.join(this.outDir_, 'SUCCEED');
    this.lastSucceedChangeset_ = null;
    this.latestChangeset_ = null;

    this.childResult_ = {};

    // Upload server
    this.remoteSshHost_ = null;
    this.remoteDir_ = null;
    this.remoteSshDir_ = null;

    // Config emial service
    this.server_ = null;
    this.message_ = null;
    this.subject_ = null;
    this.test_ = null;
  }

  /**
   * Init config_ member.
   * @param {object} options An object containing options as key-value pairs
   *    {backend:"", config:"",}.
   */
  init(options) {
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

    if (!this.config_.cleanBuild) {
      try {
        this.lastSucceedChangeset_ = fs.readFileSync(
            this.lastSucceedChangesetFile_, 'utf8');
        this.config_.logger.debug(
            `Last sucessful build changeset is ${this.lastSucceedChangeset_}`);
      } catch (e) {
        this.config_.logger.info('Not found last sucessful build.');
      }
    }

    if (!this.config_.archiveServer.host ||
      !this.config_.archiveServer.dir ||
      !this.config_.archiveServer.sshUser) {
      this.config_.logger.info(
          `Insufficient archive-server settings configure in ${configFile}`);
      return;
    }

    this.remoteSshHost_ = this.config_.archiveServer.sshUser + '@' +
      this.config_.archiveServer.host;

    if (!this.config_.emailService.user ||
        !this.config_.emailService.host ||
        !this.config_.emailService.from ||
        !this.config_.emailService.to) {
      this.config_.logger.info(
          `Insufficient email-service settings configure in ${configFile}`);
      return;
    }

    this.server_ = email.server.connect({
      user: this.config_.emailService.user,
      host: this.config_.emailService.host});
    this.message_ = {
      from: this.config_.emailService.from,
      to: this.config_.emailService.to,
      subject: this.subject_,
      text: this.test_,
    };
  };

  /**
   * Run specified command with optional options.
   * @param {string} cmd Command string.
   * @param {object} options An object containing options as key-value pairs
   *    {backend:"", config:"",}.
   */
  async run(cmd, options) {
    this.init(options);

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
   * Package libraries and executable files.
   */
  async actionPackage() {
    this.config_.logger.info('Action package');
    const packageFile = path.join(this.rootDir_, this.config_.packageName);

    switch (this.config_.targetOs) {
      case 'linux':
        // Compress files
        await this.childCommand('tar',
            ['czf', packageFile, '-C', this.rootDir_, 'out/Shared'],
            this.rootDir_);
        break;
      case 'win':
        // TODO
        // await this.childCommand(path.join(this.rootDir_,
        //   'third_party', 'lzma_sdk', 'Executable', '7za.exe'),
        //   ['x', '-y', '-sdel', packageFile], this.outDir_);
        // // Zip files
        // await this.childCommand(path.join(__dirname, 'make_win32_zip.bat'),
        //   [this.rootDir_, __dirname], this.outDir_);
        break;
      default:
        break;
    }
  }

  /**
   * Upload package and log file to stored server.
   */
  async actionUpload() {
    this.config_.logger.info('Action upload');
    if (!this.remoteSshHost_) return;

    // if (this.lastSucceedChangeset_ === this.latestChangeset_) {
    //   this.config_.logger.info(
    //     'No change since last sucessful build, skip this time.');
    //   return;
    // }
    const packageFile = path.join(this.rootDir_, this.config_.packageName);
    try {
      fs.accessSync(packageFile);
    } catch (e) {
      this.config_.logger.error(`Fail to access ${packagedFile}`);
      return;
    }

    await this.makeRemoteDir();
    await this.childCommand(
        'scp', [packageFile, this.remoteSshDir_], this.rootDir_);

    const md5Content = crypto.createHash('md5')
        .update(fs.readFileSync(packageFile)).digest('hex');
    const md5File = packageFile + '.md5';

    fs.writeFile(md5File, md5Content, (err) => {
      if (err) throw err;

      this.childCommand(
          'scp', [md5File, this.remoteSshDir_], this.rootDir_);
    });

    await this.uploadLogfile();
  }

  /**
   * Do notify by sending email.
   */
  async actionNotify() {
    this.config_.logger.info('Action notify');
    // TODO
  }

  /**
   * Create remote directory
   */
  async makeRemoteDir() {
    if (!this.remoteSshHost_) return;

    this.remoteDir_ = path.join(
        this.config_.archiveServer.dir,
        this.latestChangeset_.substring(0, 7),
        this.config_.targetOs + '_' + this.config_.targetCpu);
    let success = false;
    try {
      fs.accessSync(this.lastSucceedChangesetFile_);
      success = true;
    } catch (e) {
      success = false;
    }
    this.remoteDir_ += success ? '_SUCCEED': '_FAILED';
    this.remoteSshDir_ = this.remoteSshHost_ + ':' + this.remoteDir_ + '/';
    let dir = '';
    if (os.platform() == 'win32') {
      dir = this.remoteDir_.split(path.sep).join('/');
    }
    this.config_.logger.info(`HEAD is at ${dir}`);

    await this.childCommand('ssh',
        [this.remoteSshHost_, 'mkdir', '-p', this.remoteDir_], this.rootDir_);
  }

  /**
   * Upload log file
   */
  async uploadLogfile() {
    if (!this.remoteSshHost_) return;

    await this.makeRemoteDir();
    await this.childCommand(
        'scp', [this.config_.logFile, this.remoteSshDir_], this.rootDir_);
  }

  /**
   * Send status email
   * @param {Boolean} success build status
   * @param {String} cause cause string
   * @return {object} promise.
   */
  sendEmail(success, cause) {
    return new Promise((resolve, reject) => {
      this.message_.subject = 'WebNN nightly build successfully';

      if (!success) this.message_.subject = 'WebNN nightly build failed';

      this.message_.text = 'Hi all,\n \nThe status of WebNN nightly-build on ' +
        this.config_.targetOs + '(' + this.config_.targetCpu + ') is:\n';
      if (!success) {
        this.message_.text += 'Failed(' + cause + ')\n \nThanks';
      } else {
        this.message_.text += cause + '\n \nThanks';
      }

      this.server_.send(this.message_, function(err, message) {
        resolve(message['text']);
      });
    });
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
