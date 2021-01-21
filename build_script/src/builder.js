// Use of this source code is governed by an Apache 2.0 license
// that can be found in the LICENSE file.

'use strict';

const fs = require('fs');
const path = require('path');
const rimraf = require('rimraf');
const {spawn} = require('child_process');
const crypto = require('crypto');
const os = require('os');
// const email = require('emailjs');

/**
 * Builder class.
 */
class Builder {
  /**
   * @param {BuilderConf} conf configuration file.
   */
  constructor(conf) {
    this.conf_ = conf;
    this.supportedActions_ = ['sync', 'build', 'package', 'upload', 'all'];

    // Get last sucessful changeset
    this.lastSucceedChangesetFile_ = path.join(this.conf_.outDir, 'SUCCEED');
    this.lastSucceedChangeset_ = null;
    this.latestChangeset_ = null;
    if (!this.conf_.cleanBuild) {
      try {
        this.lastSucceedChangeset_ = fs.readFileSync(
            this.lastSucceedChangesetFile_, 'utf8');
        this.conf_.logger.debug(
            `Last sucessful build changeset is ${this.lastSucceedChangeset_}`);
      } catch (e) {
        this.conf_.logger.info('Not found last sucessful build.');
      }
    }

    this.childResult_ = {};

    // // Upload server
    // this.remoteSshHost_ = null;
    // this.remoteDir_ = null;
    // this.remoteSshDir_ = null;

    // if (!this.conf_.archiveServer.host ||
    //     !this.conf_.archiveServer.dir ||
    //     !this.conf_.archiveServer.sshUser) {
    //   this.conf_.logger.info(
    //       'Insufficient archive-server given in ' + this.conf_.confFile);
    //   return;
    // }

    // this.remoteSshHost_ =
    //   this.conf_.archiveServer.sshUser + '@' + this.conf_.archiveServer.host;

    // // Config emial service
    // this.server_ = null;
    // this.message_ = null;
    // this.subject_ =null;
    // this.test_ = null;

    // if (!this.conf_.emailService.user ||
    //     !this.conf_.emailService.host ||
    //     !this.conf_.emailService.from ||
    //     !this.conf_.emailService.to) {
    //   this.conf_.logger.info(
    //       'Insufficient email-service given in ' + this.conf_.confFile);
    //   return;
    // }

    // this.server_ = email.server.connect(
    //     {
    //       user: this.conf_.emailService.user,
    //       host: this.conf_.emailService.host,
    //     });
    // this.message_ = {
    //   from: this.conf_.emailService.from,
    //   to: this.conf_.emailService.to,
    //   subject: this.subject_,
    //   text: this.test_,
    // };
  }

  /**
   * return {string} supported actions.
   */
  get supportedActions() {
    return this.supportedActions_.toString();
  }

  /**
   * Run command.
   * @param {string} action command.
   */
  async run(action) {
    this.conf_.logger.debug('Action: ' + action);
    await this.updateChangeset();

    // skip if sync action is not include and changesets are same
    if ((action !== 'sync' || action !== 'all') &&
        (this.lastSucceedChangeset_ === this.latestChangeset_)) {
      this.conf_.logger.info(
          'No change since last sucessful build, skip this time.');
      return;
    }

    switch (action) {
      case 'sync':
        await this.actionSync();
        await this.updateChangeset();
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
      case 'all':
        await this.actionSync();
        await this.updateChangeset();
        await this.actionBuild();
        // await this.actionPackage();
        // await this.actionUpload();
        break;
      default:
        this.conf_.logger.error('Unsupported action %s', action);
        process.exit(1);
    }

    // const emailMessage = await this.sendEmail(
    //     this.childResult_.success, 'Successful');
    // this.conf_.logger.info(emailMessage);
  }

  /**
   * Run 'gclient sync' command
   */
  async actionSync() {
    this.conf_.logger.info('Action sync');
    await this.childCommand('git', ['pull', '--rebase'], this.conf_.rootDir);
    await this.childCommand('gclient', ['sync'], this.conf_.rootDir);

    if (!this.childResult_.success) {
      const emailMessage = await this.sendEmail(
          this.childResult_.success, 'Run \'gclient sync\' command failed');
      this.conf_.logger.error(emailMessage);
      await this.uploadLogfile();
      process.exit(1);
    }
  }

  /**
   * Run 'gn gen' and 'ninja -C' commands
   */
  async actionBuild() {
    this.conf_.logger.info('Action build');

    if (this.conf_.cleanBuild) {
      try {
        rimraf.sync(this.conf_.outDir);
      } catch (e) {
        this.conf_.logger.error(e);
      }
    }

    let genCmd = ['gen', `--args=${this.conf_.gnArgs}`, this.conf_.outDir];

    if (this.conf_.backend === 'wind') {
      genCmd =
        ['gen', `--ide=vs --args=${this.conf_.gnArgs}`, this.conf_.outDir];
    }

    await this.childCommand('gn', genCmd, this.conf_.rootDir);

    if (!this.childResult_.success) {
      const emailMessage = await this.sendEmail(
          this.childResult_.success, 'Run \'gn gen\' command failed');
      this.conf_.logger.error(emailMessage);
      await this.uploadLogfile();
      process.exit(1);
    }

    // Remove SUCCEED file if changeset different
    if (fs.existsSync(this.lastSucceedChangesetFile_) &&
        (this.lastSucceedChangeset_ !== this.latestChangeset_)) {
      try {
        fs.unlinkSync(this.lastSucceedChangesetFile_);
        this.lastSucceedChangeset_ = null;
      } catch (e) {
        this.conf_.logger.error(e);
      }
    }

    const args = ['-C', this.conf_.outDir]; // .concat(this.conf_.buildTargets)
    await this.childCommand(
        'ninja', args, this.conf_.rootDir);
    if (!this.childResult_.success) {
      // const emailMessage = await this.sendEmail(
      //     this.childResult_.success, 'Run \'ninja -C\' command failed');
      // this.conf_.logger.error(emailMessage);
      // await this.uploadLogfile();
      process.exit(1);
    } else {
      fs.writeFileSync(this.lastSucceedChangesetFile_, this.latestChangeset_);
    }
  }

  /**
   * Run 'package' command
   */
  async actionPackage() {
    this.conf_.logger.info('Action package');

    // Zip files
    switch (this.conf_.targetOs) {
      case 'linux':
        // TODO
        break;
      case 'win':
        // TODO
        break;
      default:
        break;
    }
  }

  /**
   * Run 'upload' command
   */
  async actionUpload() {
    this.conf_.logger.info('Action upload');
    if (!this.remoteSshHost_) return;

    if (this.lastSucceedChangeset_ === this.latestChangeset_) {
      this.conf_.logger.info(
          'No change since last sucessful build, skip this time.');
      return;
    }

    try {
      fs.accessSync(this.conf_.packagedFile);
    } catch (e) {
      this.conf_.logger.error('Fail to access ' + this.conf_.packagedFile);
      return;
    }

    await this.makeRemoteDir();
    await this.childCommand('scp',
        [this.conf_.packagedFile, this.remoteSshDir_], this.conf_.rootDir);

    const md5Content = crypto.createHash('md5')
        .update(fs.readFileSync(this.conf_.packagedFile)).digest('hex');
    const md5File = this.conf_.packagedFile + '.md5';

    fs.writeFile(md5File, md5Content, (err) => {
      if (err) throw err;

      this.childCommand(
          'scp', [md5File, this.remoteSshDir_], this.conf_.rootDir);
    });

    await this.uploadLogfile();
  }

  /**
   * Get latest changeset
   */
  async updateChangeset() {
    const obj = {};
    await this.childCommand('git', ['pull', '--rebase'], this.conf_.rootDir);
    await this.childCommand(
        'git', ['rev-parse', 'HEAD'], this.conf_.rootDir, obj);
    this.latestChangeset_ = obj.changeset;
    this.conf_.logger.info(`HEAD is at ${this.latestChangeset_}`);
  }

  /**
   * Create remote directory
   */
  async makeRemoteDir() {
    if (!this.remoteSshHost_) return;

    this.remoteDir_ = path.join(
        this.conf_.archiveServer.dir,
        this.latestChangeset_.substring(0, 7),
        this.conf_.targetOs + '_' + this.conf_.targetCpu);
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
    this.conf_.logger.info(`HEAD is at ${dir}`);

    await this.childCommand(
        'ssh', [this.remoteSshHost_, 'mkdir', '-p', dir], this.conf_.rootDir);
  }

  /**
   * Upload log file
   */
  async uploadLogfile() {
    if (!this.remoteSshHost_) return;

    await this.makeRemoteDir();
    await this.childCommand(
        'scp', [this.conf_.logFile, this.remoteSshDir_], this.conf_.rootDir);
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
        this.conf_.targetOs + '(' + this.conf_.targetCpu + ') is:\n';
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
      this.conf_.logger.info('Execute command: ' + cmdFullStr);

      // const child = spawn(cmd, [...args], {cwd: this.conf_.rootDir});
      const child = spawn(cmd, [...args], {cwd: cwd});

      child.stdout.on('data', (data) => {
        if (result) result.changeset = data.toString();
        this.conf_.logger.debug(`${data.toString()}`);
      });

      child.stderr.on('data', (data) => {
        this.conf_.logger.error(data.toString());
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
