// https://github.com/import-js/eslint-plugin-import#rules
module.exports = {
  extends: ['plugin:import/recommended', 'plugin:import/typescript'],
  settings: {
    'import/resolver': {
      typescript: {
        project: 'lpm_frontend/tsconfig.json'
      }
    }
  }
};
