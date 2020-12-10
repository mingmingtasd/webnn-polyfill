import fs from "fs";
import nunjucks from "nunjucks";

import pkg from "../../package.json";

import {
  warn
} from "../utils.mjs";

let ast = null;

const GYP_TEMPLATE = fs.readFileSync(`${pkg.config.TEMPLATE_DIR}/binding-gyp.njk`, "utf-8");
var DAWN_PATH = fs.readFileSync(pkg.config.DAWN_PATH, "utf-8").trim();
DAWN_PATH = DAWN_PATH.replace(/\\/g, "/");

nunjucks.configure({ autoescape: true });

export default function() {
  let out = {};
  let vars = {
    DAWN_PATH,
    SOURCE_INCLUDES: [
      "src/*.cpp",
      "src/ops/*.cc"
    ].map(v => `"${v}"`)
  };
  // binding.gyp
  {
    let template = GYP_TEMPLATE;
    let output = nunjucks.renderString(template, vars);
    out.gyp = output;
  }
  return out;
};
