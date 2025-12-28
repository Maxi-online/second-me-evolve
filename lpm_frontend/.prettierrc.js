// https://prettier.io/docs/en/options.html
// https://github.com/trivago/prettier-plugin-sort-imports

module.exports = {
  trailingComma: 'none',
  singleQuote: true,
  printWidth: 100,
  importOrder: ['<THIRD_PARTY_MODULES>', '^@/(.*)$', '^[./]'],
  importOrderSeparation: true,
  importOrderSortSpecifiers: true,
  semi: true
};
