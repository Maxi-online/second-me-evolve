// https://stylelint.io/user-guide/rules
module.exports = {
  extends: [
    'stylelint-config-standard',
    'stylelint-config-rational-order',
    'stylelint-prettier/recommended'
  ],
  overrides: [
    {
      files: '**/*.less',
      customSyntax: 'postcss-less'
    }
  ],
  rules: {
    'selector-class-pattern': null,
    'color-function-notation': 'legacy',
    'declaration-block-no-redundant-longhand-properties': null,
    'selector-no-vendor-prefix': [
      true,
      {
        ignoreSelectors: ['input-placeholder']
      }
    ],
    'property-no-vendor-prefix': [
      true,
      {
        ignoreProperties: ['user-select', 'line-clamp', 'appearance']
      }
    ],
    'selector-pseudo-class-no-unknown': [
      true,
      {
        ignorePseudoClasses: ['global']
      }
    ],
    'unit-no-unknown': [
      true,
      {
        ignoreUnits: ['/^rpx$/']
      }
    ]
  }
};
