module.exports = {
  plugins: {
    'posthtml-expressions': {
      locals: {
        process: {
          env: process.env
        }
      }
    }
  }
};