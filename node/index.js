const fs = require("fs");
const path = require("path");

const pkg = require("./package.json");

let {platform} = process;

const dawnVersion = "0.0.1";

const bindingsPath = path.join(__dirname, `${pkg.config.GEN_OUT_DIR}/`);
const generatedPath = bindingsPath + `${dawnVersion}/${platform}`;

module.exports = require(`${generatedPath}/build/Release/addon-${platform}.node`);
