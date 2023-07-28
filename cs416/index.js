$(document).ready(function() {
  var interval = 1000; // Change image every 5 seconds
  var $slideshow = $('.slideshow');
  var $images = $slideshow.find('img');
  var currentImageIndex = 0;
  var totalImages = $images.length;
  var yearSlider = document.getElementById('yearSlider');
  var yearLabel = document.getElementById('yearLabel');

  // Function to change the image
  //function changeImage() {
    //  $images.removeClass('active');
      //currentImageIndex = (currentImageIndex + 1) % totalImages;
      //$images.eq(currentImageIndex).addClass('active');
  //} 
    function showImageByYear(year) {
      $images.removeClass('active'); // Hide all images
      var imageIndex = year - 1993; // Calculate the image index based on the year
      $images.eq(imageIndex).addClass('active'); // Show the image for the specified year
}
    yearSlider.addEventListener('input', function() {
      var selectedYear = parseInt(yearSlider.value, 10);
      yearLabel.innerText = selectedYear;
      showImageByYear(selectedYear);
});

    

  // Start the slideshow
 // var slideshowInterval = setInterval(changeImage, interval);
  //yearSlider.addEventListener('input', function() {
    //var selectedYear = parseInt(yearSlider.value, 10);
    //showImageByYear(selectedYear);
//});

   showImageByYear(parseInt(yearSlider.value, 10));
  // Pause and play the slideshow on button click
 // $('#playPauseButton').click(function() {
   //   if ($(this).hasClass('playing')) {
     //     clearInterval(slideshowInterval);
       //   $(this).removeClass('playing').text('Play');
      //} else {
        //  slideshowInterval = setInterval(changeImage, interval);
          //$(this).addClass('playing').text('Pause');
      //}
 // });
});

