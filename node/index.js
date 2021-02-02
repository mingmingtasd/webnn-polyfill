const fs = require("fs");
const path = require("path");

const pkg = require("./package.json");

let {platform} = process;

module.exports = require('bindings')(`addon-${platform}.node`);
